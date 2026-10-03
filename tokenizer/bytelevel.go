package tokenizer

var byteToCharacter, characterToByte = makeByteLevelTables()

func byteIsPrintedAsItself(value int) bool {
	return (value >= '!' && value <= '~') || (value >= 0xA1 && value <= 0xAC) || (value >= 0xAE && value <= 0xFF)
}

func makeByteLevelTables() ([256]rune, map[rune]byte) {
	var toCharacter [256]rune
	toByte := map[rune]byte{}
	nextUnusedCharacter := 256
	for value := 0; value < 256; value++ {
		character := rune(value)
		if !byteIsPrintedAsItself(value) {
			character = rune(nextUnusedCharacter)
			nextUnusedCharacter++
		}
		toCharacter[value] = character
		toByte[character] = byte(value)
	}
	return toCharacter, toByte
}

func bytesToByteLevelText(text string) string {
	characters := make([]rune, 0, len(text))
	for i := 0; i < len(text); i++ {
		characters = append(characters, byteToCharacter[text[i]])
	}
	return string(characters)
}

func byteLevelTextToBytes(byteLevelText string) []byte {
	var result []byte
	for _, character := range byteLevelText {
		value, found := characterToByte[character]
		if !found {
			result = append(result, []byte(string(character))...)
			continue
		}
		result = append(result, value)
	}
	return result
}
