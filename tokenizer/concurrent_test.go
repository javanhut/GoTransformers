package tokenizer

import (
	"reflect"
	"sync"
	"testing"
)

func TestEncodeFromManyGoroutines(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog, then the dog chases the fox around the yard again and again"
	textTokenizer := Train(text, 300)
	want := Train(text, 300).Encode(text)
	var waitGroup sync.WaitGroup
	for worker := range 8 {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			for repeat := range 50 {
				piece := text[(worker+repeat)%10:]
				if got := textTokenizer.Encode(piece); len(got) == 0 {
					t.Errorf("worker %d got no tokens", worker)
				}
			}
			if got := textTokenizer.Encode(text); !reflect.DeepEqual(got, want) {
				t.Errorf("worker %d encoded differently", worker)
			}
		}()
	}
	waitGroup.Wait()
}
