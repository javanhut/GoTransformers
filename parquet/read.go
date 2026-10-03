package parquet

func (opened *File) ReadStringColumn(rowGroupIndex int, columnName string) ([]string, error) {
	decoded, err := opened.readColumn(rowGroupIndex, columnName, textKind)
	if err != nil {
		return nil, err
	}
	if decoded.isPresent == nil {
		return decoded.values.texts, nil
	}
	texts := make([]string, len(decoded.isPresent))
	valueIndex := 0
	for row := 0; row < len(decoded.isPresent); row++ {
		if decoded.isPresent[row] {
			texts[row] = decoded.values.texts[valueIndex]
			valueIndex++
		}
	}
	return texts, nil
}

func (opened *File) ReadInt64Column(rowGroupIndex int, columnName string) ([]int64, error) {
	decoded, err := opened.readColumn(rowGroupIndex, columnName, integerKind)
	if err != nil {
		return nil, err
	}
	if decoded.isPresent == nil {
		return decoded.values.integers, nil
	}
	integers := make([]int64, len(decoded.isPresent))
	valueIndex := 0
	for row := 0; row < len(decoded.isPresent); row++ {
		if decoded.isPresent[row] {
			integers[row] = decoded.values.integers[valueIndex]
			valueIndex++
		}
	}
	return integers, nil
}

func (opened *File) ReadFloat64Column(rowGroupIndex int, columnName string) ([]float64, error) {
	decoded, err := opened.readColumn(rowGroupIndex, columnName, floatKind, integerKind)
	if err != nil {
		return nil, err
	}
	presentFloats := decoded.values.floats
	if decoded.values.integers != nil {
		presentFloats = make([]float64, len(decoded.values.integers))
		for index := 0; index < len(decoded.values.integers); index++ {
			presentFloats[index] = float64(decoded.values.integers[index])
		}
	}
	if decoded.isPresent == nil {
		return presentFloats, nil
	}
	floats := make([]float64, len(decoded.isPresent))
	valueIndex := 0
	for row := 0; row < len(decoded.isPresent); row++ {
		if decoded.isPresent[row] {
			floats[row] = presentFloats[valueIndex]
			valueIndex++
		}
	}
	return floats, nil
}

func (opened *File) ReadBoolColumn(rowGroupIndex int, columnName string) ([]bool, error) {
	decoded, err := opened.readColumn(rowGroupIndex, columnName, booleanKind)
	if err != nil {
		return nil, err
	}
	if decoded.isPresent == nil {
		return decoded.values.booleans, nil
	}
	booleans := make([]bool, len(decoded.isPresent))
	valueIndex := 0
	for row := 0; row < len(decoded.isPresent); row++ {
		if decoded.isPresent[row] {
			booleans[row] = decoded.values.booleans[valueIndex]
			valueIndex++
		}
	}
	return booleans, nil
}

func (opened *File) ReadNullColumn(rowGroupIndex int, columnName string) ([]bool, error) {
	decoded, err := opened.readColumn(rowGroupIndex, columnName, textKind, integerKind, floatKind, booleanKind)
	if err != nil {
		return nil, err
	}
	if decoded.isPresent == nil {
		return make([]bool, decoded.values.length()), nil
	}
	isNull := make([]bool, len(decoded.isPresent))
	for row := 0; row < len(decoded.isPresent); row++ {
		isNull[row] = !decoded.isPresent[row]
	}
	return isNull, nil
}

func (opened *File) ForEachRowGroupOfStrings(columnName string, handle func(rowGroupIndex int, texts []string) error) error {
	for rowGroupIndex := 0; rowGroupIndex < opened.NumberOfRowGroups(); rowGroupIndex++ {
		texts, err := opened.ReadStringColumn(rowGroupIndex, columnName)
		if err != nil {
			return err
		}
		if err := handle(rowGroupIndex, texts); err != nil {
			return err
		}
	}
	return nil
}

func (opened *File) ReadAllStrings(columnName string) ([]string, error) {
	var allTexts []string
	err := opened.ForEachRowGroupOfStrings(columnName, func(rowGroupIndex int, texts []string) error {
		allTexts = append(allTexts, texts...)
		return nil
	})
	return allTexts, err
}

func (opened *File) ReadAllInt64s(columnName string) ([]int64, error) {
	var allIntegers []int64
	for rowGroupIndex := 0; rowGroupIndex < opened.NumberOfRowGroups(); rowGroupIndex++ {
		integers, err := opened.ReadInt64Column(rowGroupIndex, columnName)
		if err != nil {
			return nil, err
		}
		allIntegers = append(allIntegers, integers...)
	}
	return allIntegers, nil
}

func (opened *File) ReadAllFloat64s(columnName string) ([]float64, error) {
	var allFloats []float64
	for rowGroupIndex := 0; rowGroupIndex < opened.NumberOfRowGroups(); rowGroupIndex++ {
		floats, err := opened.ReadFloat64Column(rowGroupIndex, columnName)
		if err != nil {
			return nil, err
		}
		allFloats = append(allFloats, floats...)
	}
	return allFloats, nil
}
