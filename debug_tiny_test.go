package chess

import (
    "fmt"
    "io"
    "testing"
)

func TestDebugTiny(t *testing.T) {
    path := "/tmp/test_tiny.binpack"
    is, _ := isBINPFormat(path)
    if !is { t.Fatal("Not BINP") }
    reader, err := openBINPReader(path)
    if err != nil { t.Fatal(err) }
    defer reader.Close()
    count := 0
    for {
        s, err := reader.Next()
        if err == io.EOF { break }
        if err != nil {
            fmt.Printf("  ERROR: %v\n", err)
            break
        }
        if s == nil { continue }
        count++
        if count <= 20 {
            fmt.Printf("  [%d] score=%.0f result=%.0f\n", count, s.Score, s.Result)
        }
    }
    fmt.Printf("Total: %d, Stems: %d, MT_OK: %d, MT_FAIL: %d\n",
        count, reader.StemsRead, reader.MovetextDecodeSuccess, reader.MovetextDecodeFailures)
}
