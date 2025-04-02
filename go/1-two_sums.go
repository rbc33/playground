package main

import "fmt"

func twoSum(nums []int, target int) []int {
	hashMap := make(map[int]int)
	res := []int{0, 0}

	for i := 0; i < len(nums); i++ {

		value, exist := hashMap[target-nums[i]]
		if exist == true {
			res[0] = value
			res[1] = i
			return res
		} else {
			key := nums[i]
			hashMap[key] = i
		}
	}
	return res

}
func main() {
    nums := []int{1, 2, 4, 5}  // Array literal con llaves {}
    res := twoSum(nums, 9)
    fmt.Printf("%v\n", res)    // Printf con mayúscula y formato %v para slice
}