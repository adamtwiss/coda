#!/usr/bin/env python3
"""Training-curve inspector — central on Hercules, SSH-fetches Bullet log.txt
from the GPU hosts and renders interactive loss-decay overlays.

Run:  python3 scripts/train_curve_app.py            # serves on :8042
      python3 scripts/train_curve_app.py --port 9000 --hosts gpu0,gpu2,gpu4

Then open http://localhost:8042/ (or tunnel the port).

log.txt format on the hosts is `superbatch,batch,loss` per line. Snapshot dirs
`<net-id>-<SB>` are checkpoints of ONE run; the largest-SB dir's log.txt holds
the whole run. `<net-id>-<SB>-swa` is the SWA-averaged twin (same training log).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

REMOTE_CKPT = "~/code/bullet/checkpoints"
CACHE_DIR = "/tmp/traincurve_cache"
INDEX_TTL = 90  # seconds

DIR_RE = re.compile(r"^(?P<base>.+)-(?P<sb>\d+)(?P<swa>-swa)?$")

_index_lock = threading.Lock()
_index_cache = {"ts": 0.0, "data": None}


# Cap concurrent ssh subprocesses. ThreadingHTTPServer spawns an unbounded
# thread per request and each sh() forks an ssh with its own pipes/FDs; when a
# vast.ai host lags (timeouts up to 120s) under browser auto-polling, FDs pile
# up until serve_forever's accept() raises "Too many open files" and the whole
# process dies. Bounding concurrency is the structural guard against that.
_ssh_sem = threading.Semaphore(6)


def sh(host, cmd, timeout=40):
    """Run a remote command, return stdout (str). Strips vast.ai banner lines.

    Never raises: a slow/unreachable host degrades to "" so a single down
    vast.ai instance can't propagate an exception up through build_index into
    the server loop. Errors are logged to stderr for diagnosis."""
    try:
        with _ssh_sem:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=12", "-o", "BatchMode=yes", host, cmd],
                capture_output=True, text=True, timeout=timeout,
            ).stdout
    except Exception as e:
        sys.stderr.write(f"[sh] {host}: {type(e).__name__}: {e}\n")
        sys.stderr.flush()
        return ""
    keep = [l for l in out.splitlines()
            if not re.search(r"vast\.ai|Have fun|authentication", l, re.I)]
    return "\n".join(keep)


def decode_id(base):
    """Decode a net-id into recipe deviations from canonical.
    Canonical = wdl0.15, s800, warm30, factoriser, kb reckless, crelu, seed42,
    FT default(768), no fenskip, no SWA, sequential data order."""
    d = {}
    m = re.search(r"ft(\d+)", base)
    d["ft"] = int(m.group(1)) if m else 768
    m = re.search(r"l1[-_]?(\d+)", base)
    d["l1"] = int(m.group(1)) if m else 16
    m = re.search(r"\bw(\d+)\b", base)
    d["wdl"] = (int(m.group(1)) / 100.0) if m else 0.15
    m = re.search(r"swa(\d+)", base)
    d["swa_start"] = int(m.group(1)) if m else None
    m = re.search(r"fs([0-9.]+)", base)
    d["fenskip"] = float(m.group(1)) if m else 0.0
    if "inter" in base:
        d["order"] = "interleave"
    elif "mix" in base or d["fenskip"]:
        d["order"] = "seq+fenskip"
    else:
        d["order"] = "sequential"
    return d


def build_index():
    """Group all checkpoint dirs by run. Returns list of run dicts."""
    runs = []
    for host in HOSTS:
        listing = sh(host, f"ls -d {REMOTE_CKPT}/*/ 2>/dev/null")
        groups = {}  # base -> {sbs, swa_sbs}
        for line in listing.splitlines():
            name = line.rstrip("/").split("/")[-1]
            m = DIR_RE.match(name)
            if not m:
                continue
            base, sb = m.group("base"), int(m.group("sb"))
            g = groups.setdefault(base, {"sbs": set(), "swa": set()})
            (g["swa"] if m.group("swa") else g["sbs"]).add(sb)
        for base, g in groups.items():
            all_sbs = g["sbs"] | g["swa"]
            if not all_sbs:
                continue
            max_sb = max(all_sbs)
            # prefer a non-swa dir as the log source (same log either way)
            src_sb = max(g["sbs"]) if g["sbs"] else max_sb
            src_swa = "" if (g["sbs"] and src_sb == max(g["sbs"])) else "-swa"
            runs.append({
                "host": host,
                "base": base,
                "max_sb": max_sb,
                "snapshots": sorted(all_sbs),
                "has_swa": bool(g["swa"]),
                "src_dir": f"{base}-{src_sb}{src_swa}",
                "decode": decode_id(base),
            })
    runs.sort(key=lambda r: (r["host"], r["base"]))
    return runs


def get_index(force=False):
    with _index_lock:
        if (not force and _index_cache["data"] is not None
                and time.time() - _index_cache["ts"] < INDEX_TTL):
            return _index_cache["data"]
        data = build_index()
        _index_cache.update(ts=time.time(), data=data)
        return data


# Server-side per-SB aggregate: reads the 152k-line log on the host and emits
# only ~800 rows `sb,mean_train,mean_val`. Keeps the slow vast.ai network hop
# tiny. Train loss is column $3 in BOTH the 3-col (sb,batch,loss) and 4-col
# (sb,batch,loss,val_loss) formats, so NF>=3 covers both. The val column ($4)
# is sparse (only rows where validation ran) — accumulated separately and left
# blank for SBs/old logs without any validation point.
AWK_AGG = (
    "awk -F, 'NF>=3{s[$1]+=$3;c[$1]++;if($1>m)m=$1;"
    "if(NF>=4 && $4!=\"\"){vs[$1]+=$4;vc[$1]++}}"
    " END{for(i=1;i<=m;i++) if(c[i]>0){"
    "v=(vc[i]>0)?sprintf(\"%.8f\",vs[i]/vc[i]):\"\";"
    "printf \"%d,%.8f,%s\\n\",i,s[i]/c[i],v}}'"
)


def fetch_series(host, base, src_dir):
    """SSH-aggregate the run's log.txt (mtime-cached) to per-SB mean loss.
    Only the ~800-row aggregate crosses the network, not the 3MB raw log."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    remote = f"{REMOTE_CKPT}/{src_dir}/log.txt"
    mtime = sh(host, f"stat -c %Y {remote} 2>/dev/null").strip() or "0"
    tag = f"{host}__{base}".replace("/", "_")
    parsed = os.path.join(CACHE_DIR, f"{tag}.{mtime}.json")
    if os.path.exists(parsed):
        with open(parsed) as f:
            return json.load(f)
    agg = sh(host, f"{AWK_AGG} {remote} 2>/dev/null", timeout=120)
    sbs, mean, val = [], [], []
    for line in agg.splitlines():
        p = line.split(",")
        if len(p) < 2:
            continue
        try:
            sbs.append(int(p[0])); mean.append(float(p[1]))
            val.append(float(p[2]) if len(p) >= 3 and p[2] != "" else None)
        except ValueError:
            continue
    out = {
        "host": host, "base": base,
        "sb": sbs,
        "mean": mean,
        "val": val,
        "has_val": any(v is not None for v in val),
        "final": mean[-1] if mean else None,
    }
    # prune stale parsed caches for this tag
    for f in os.listdir(CACHE_DIR):
        if f.startswith(tag + ".") and f.endswith(".json"):
            os.remove(os.path.join(CACHE_DIR, f))
    with open(parsed, "w") as f:
        json.dump(out, f)
    return out


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Coda training curves</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font:13px/1.4 system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:340px;border-right:1px solid #ccc;overflow:auto;padding:8px;box-sizing:border-box}
 #main{flex:1;display:flex;flex-direction:column;overflow:auto}
 .run{padding:3px 4px;border-bottom:1px solid #eee;cursor:pointer}
 .run:hover{background:#f3f6ff}
 .run.sel{background:#dde9ff}
 .meta{color:#666;font-size:11px}
 .host{display:inline-block;width:42px;color:#0a7;font-weight:600}
 .swa{color:#c60;font-weight:600}
 #plots>div{height:46vh;border-bottom:1px solid #eee}
 .bar{padding:6px;background:#fafafa;border-bottom:1px solid #ccc}
 button{font:12px sans-serif;margin-right:6px}
 input#filter{width:200px}
 select{font:12px sans-serif}
</style></head><body>
<div id="side">
 <div class="bar">
  <input id="filter" placeholder="filter runs (substring)">
  <button onclick="refresh(true)">reload index</button>
  <div style="margin-top:4px"><span id="count"></span> runs</div>
 </div>
 <div id="list"></div>
</div>
<div id="main">
 <div class="bar">
  view:
  <select id="view" onchange="render()">
   <option value="mean">per-SB mean loss (log-y)</option>
   <option value="dloss">d(loss)/dSB (smoothed)</option>
   <option value="absd">|Δloss|/SB rolling (SWA-fuel proxy)</option>
   <option value="tail">tail zoom (last 25%)</option>
  </select>
  val smooth:
  <select id="smooth" onchange="render()">
   <option value="1">off</option>
   <option value="5">5 SB</option>
   <option value="11" selected>11 SB</option>
   <option value="21">21 SB</option>
   <option value="51">51 SB</option>
   <option value="101">101 SB</option>
  </select>
  <button onclick="clearSel()">clear selection</button>
  <span id="status" style="color:#888"></span>
 </div>
 <div id="plot" style="flex:1"></div>
</div>
<script>
let INDEX=[], SEL=new Map(), SERIES=new Map(), ROLL=11;
function decodeStr(d){
  let p=[];
  if(d.ft!=768)p.push("ft"+d.ft);
  if(d.l1!=16)p.push("l1="+d.l1);
  if(d.wdl!=0.15)p.push("w"+Math.round(d.wdl*100));
  if(d.order!="sequential")p.push(d.order);
  if(d.fenskip)p.push("fs"+d.fenskip);
  if(d.swa_start)p.push("swa@"+d.swa_start);
  return p.join(" ")||"canonical";
}
async function refresh(force){
  let r=await fetch("/api/index"+(force?"?force=1":""));
  INDEX=await r.json(); draw();
}
function draw(){
  let f=document.getElementById("filter").value.toLowerCase();
  let list=document.getElementById("list"); list.innerHTML="";
  let shown=0;
  for(let r of INDEX){
    let key=r.host+"|"+r.base;
    if(f && !(r.base.toLowerCase().includes(f)||r.host.includes(f)))continue;
    shown++;
    let div=document.createElement("div");
    div.className="run"+(SEL.has(key)?" sel":"");
    div.innerHTML="<span class='host'>"+r.host+"</span><b>"+r.base+"</b>"+
      "<div class='meta'>s"+r.max_sb+" &middot; "+r.snapshots.length+" snaps"+
      (r.has_swa?" &middot; <span class='swa'>SWA</span>":"")+
      " &middot; "+decodeStr(r.decode)+"</div>";
    div.onclick=()=>toggle(r,key);
    list.appendChild(div);
  }
  document.getElementById("count").textContent=shown+"/"+INDEX.length;
}
async function toggle(r,key){
  if(SEL.has(key)){SEL.delete(key);}
  else{
    SEL.set(key,r);
    if(!SERIES.has(key)){
      setStatus("fetching "+r.base+" ...");
      let u="/api/series?host="+r.host+"&base="+encodeURIComponent(r.base)+
            "&src="+encodeURIComponent(r.src_dir);
      try{let s=await (await fetch(u)).json(); SERIES.set(key,s);}
      catch(e){setStatus("fetch failed: "+e); SEL.delete(key);}
      setStatus("");
    }
  }
  draw(); render();
}
function clearSel(){SEL.clear();draw();render();}
function setStatus(s){document.getElementById("status").textContent=s;}
function rollAbs(mean){
  let d=[];for(let i=1;i<mean.length;i++)d.push(Math.abs(mean[i]-mean[i-1]));
  let out=[];for(let i=0;i<d.length;i++){
    let lo=Math.max(0,i-(ROLL>>1)),hi=Math.min(d.length,i+(ROLL>>1)+1);
    let s=0;for(let j=lo;j<hi;j++)s+=d[j];out.push(s/(hi-lo));}
  return out;
}
function smoothDeriv(mean){
  let out=[];for(let i=0;i<mean.length;i++){
    let lo=Math.max(0,i-2),hi=Math.min(mean.length-1,i+2);
    out.push((mean[hi]-mean[lo])/(hi-lo));}
  return out;
}
function smoothArr(arr,w){
  if(w<=1)return arr;
  let h=w>>1,out=new Array(arr.length);
  for(let i=0;i<arr.length;i++){
    let lo=Math.max(0,i-h),hi=Math.min(arr.length,i+h+1);
    let s=0,n=0;
    for(let j=lo;j<hi;j++)if(arr[j]!=null){s+=arr[j];n++;}
    out[i]=n>0?s/n:null;
  }
  return out;
}
function render(){
  let view=document.getElementById("view").value, traces=[], layout={};
  let sw=parseInt(document.getElementById("smooth").value)||1;
  let swTag=sw>1?" — val smoothed "+sw+" SB":"";
  for(let [key,r] of SEL){
    let s=SERIES.get(key); if(!s)continue;
    let name=r.host+":"+r.base, sb=s.sb, mean=s.mean;
    let hasVal=s.val && s.val.some(v=>v!=null);
    if(view=="mean"){
      traces.push({x:sb,y:mean,name:name,mode:"lines",line:{width:1}});
      if(hasVal){let v=smoothArr(s.val,sw);
        traces.push({x:sb,y:v,name:name+" (val)",mode:"lines",line:{width:1,dash:"dot"},connectgaps:true});}
      layout={title:"Per-SB mean loss (dotted = holdout val"+swTag+")",yaxis:{type:"log",title:"loss"},xaxis:{title:"superbatch"}};
    }else if(view=="dloss"){
      traces.push({x:sb,y:smoothDeriv(mean),name:name,mode:"lines",line:{width:1}});
      layout={title:"d(loss)/dSB (5-pt centred) — watch the ~80% bump",yaxis:{title:"d loss / dSB"},xaxis:{title:"superbatch"}};
    }else if(view=="absd"){
      traces.push({x:sb.slice(1),y:rollAbs(mean),name:name,mode:"lines",line:{width:1.2}});
      layout={title:"|Δloss|/SB ("+ROLL+"-SB rolling) = SWA-fuel proxy",yaxis:{title:"|Δ| per SB"},xaxis:{title:"superbatch"}};
    }else if(view=="tail"){
      let cut=Math.floor(sb.length*0.75);
      traces.push({x:sb.slice(cut),y:mean.slice(cut),name:name,mode:"lines+markers",marker:{size:3},line:{width:1}});
      if(hasVal){let v=smoothArr(s.val,sw);
        traces.push({x:sb.slice(cut),y:v.slice(cut),name:name+" (val)",mode:"lines+markers",marker:{size:3},line:{width:1,dash:"dot"},connectgaps:true});}
      layout={title:"Tail zoom (last 25% — dotted = holdout val"+swTag+"; does val keep falling while train bumps?)",yaxis:{title:"loss"},xaxis:{title:"superbatch"}};
    }
  }
  layout.margin={l:60,r:20,t:40,b:45};layout.legend={font:{size:10}};
  Plotly.react("plot",traces,layout,{responsive:true});
}
document.getElementById("filter").addEventListener("input",draw);
refresh(false);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/index":
                self._send(200, json.dumps(get_index(force="force" in q)))
            elif u.path == "/api/series":
                s = fetch_series(q["host"][0], q["base"][0], q["src"][0])
                self._send(200, json.dumps(s))
            else:
                self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


def main():
    global HOSTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--hosts", default="gpu3,gpu4")
    args = ap.parse_args()
    HOSTS = [h.strip() for h in args.hosts.split(",") if h.strip()]
    print(f"training-curve app: hosts={HOSTS}  http://localhost:{args.port}/",
          flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except BaseException as e:
        # Record the real cause (e.g. OSError: Too many open files) before the
        # process exits, so the supervisor log shows WHY rather than a blank.
        import traceback
        sys.stderr.write(f"serve_forever exited: {type(e).__name__}: {e}\n")
        traceback.print_exc()
        sys.stderr.flush()
        raise


HOSTS = ["gpu0", "gpu2", "gpu4"]
if __name__ == "__main__":
    main()
