//! T1-only diagnostic event tracing. No search policy changes unless a specific
//! stage/node/gate intervention is explicitly selected by environment.
use std::{cell::RefCell,fmt,sync::OnceLock};
// Explicit scalar serialization: no struct padding, pointers, or atomics copied.
pub trait State { fn put(&self, out:&mut Vec<u8>); fn get(&mut self, input:&mut &[u8]); }
macro_rules! scalar { ($($t:ty),*) => {$ (impl State for $t {
    fn put(&self,o:&mut Vec<u8>){o.extend_from_slice(&self.to_le_bytes());}
    fn get(&mut self,i:&mut &[u8]){let n=std::mem::size_of::<Self>();*self=Self::from_le_bytes(i[..n].try_into().unwrap());*i=&i[n..];}
})*}; }
scalar!(u8,u16,u32,u64,i16,i32,usize);
impl State for bool {fn put(&self,o:&mut Vec<u8>){o.push(*self as u8);}fn get(&mut self,i:&mut &[u8]){assert!(i[0]<=1);*self=i[0]!=0;*i=&i[1..];}}
impl<T:State,const N:usize> State for [T;N] {fn put(&self,o:&mut Vec<u8>){for x in self{x.put(o)}}fn get(&mut self,i:&mut &[u8]){for x in self{x.get(i)}}}
pub fn save(name:&str,data:&[u8]) {if let Ok(dir)=std::env::var("PAIR_SAVE") {std::fs::write(format!("{dir}/{name}.bin"),data).unwrap();}}
pub fn restore(name:&str)->Option<Vec<u8>> {let groups=std::env::var("PAIR_RESTORE_GROUPS").unwrap_or_default();
    if groups.split(',').any(|g|g==name) {Some(std::fs::read(format!("{}/{}.bin",std::env::var("PAIR_RESTORE").unwrap(),name)).unwrap())}else{None}}
#[derive(Default)] struct Data {stage:Option<(&'static str,i32)>, stack:Vec<(u64,i32,u64)>, next:u64}
thread_local!{static DATA:RefCell<Data>=RefCell::new(Data::default());}
pub fn target(n:u64)->bool {static AT:OnceLock<Option<u64>>=OnceLock::new();
    *AT.get_or_init(||std::env::var("PAIR_AT").ok().map(|s|s.parse().unwrap()))==Some(n)}
pub struct Stage(Option<(&'static str,i32)>);
pub fn stage(n:u64,name:&'static str,ply:i32)->Stage {DATA.with(|d|{let mut d=d.borrow_mut();let old=d.stage;
    if target(n){d.stage=Some((name,ply));}Stage(old)})}
impl Drop for Stage{fn drop(&mut self){DATA.with(|d|d.borrow_mut().stage=self.0);}}
pub fn enter(n:u64,hash:u64,ply:i32,depth:i32,alpha:i32,beta:i32,cut:bool,prior:i32)->bool {
    DATA.with(|d|{let mut d=d.borrow_mut();let Some((stage,root))=d.stage else{return false};
        d.next+=1;let id=d.next;d.stack.push((id,ply,n));
        if ply<=root+2 {eprintln!("DTRACE stage={} id={} entry={} hash={} ply={} depth={} alpha={} beta={} cut={} prior={} event=enter",stage,id,n,hash,ply,depth,alpha,beta,cut,prior);}true})
}
pub fn exit(active:bool,n:u64,value:i32,stopped:bool){if active {DATA.with(|d|{let mut d=d.borrow_mut();let(id,ply,start)=d.stack.pop().unwrap();let(stage,root)=d.stage.unwrap();
    if ply<=root+2 {eprintln!("DTRACE stage={} id={} event=exit score={} cost={} stopped={}",stage,id,value,n-start,stopped);}});}}
pub fn log(ply:i32,args:fmt::Arguments){DATA.with(|d|{let d=d.borrow();if let Some((stage,root))=d.stage {if ply<=root+2 {eprintln!("DTRACE stage={} id={} {}",stage,d.stack.last().map_or(0,|x|x.0),args);}}});}
pub fn block(gate:&str)->bool {
    static RULE:OnceLock<Option<(String,String,u64)>>=OnceLock::new();
    let rule=RULE.get_or_init(||std::env::var("PAIR_BLOCK").ok().map(|s|{let p:Vec<_>=s.split(':').collect();assert_eq!(p.len(),3);(p[0].into(),p[1].into(),p[2].parse().unwrap())}));
    DATA.with(|d|{let d=d.borrow();let Some((stage,_))=d.stage else{return false};let Some((_,_,node))=d.stack.last()else{return false};
        let yes=rule.as_ref().is_some_and(|(s,g,n)|s==stage && g==gate && n==node);
        if yes {eprintln!("BLOCK stage={} gate={} entry={}",stage,gate,node);}yes})
}
