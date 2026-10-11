import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np_arr.sum())
print("Minimum:", np_arr.min())
print("Maximum:", np_arr.max())
print("Mean:", np_arr.mean())
print("Argmax:", np_arr.argmax())

# [row: 학생, column: 과목]
scores = np.array([
    [83, 92, 100],  # 0번 학생
    [76, 63, 58],   # 1번 학생
    [91, 99, 84]    # 2번 학생
])

student_means = np.mean(scores, axis=1)
print("학생별 평균 (axis=1):")
print(student_means)

print()

subject_means = np.mean(scores, axis=0)
print("과목별 평균 (axis=0):")
print(subject_means)

scores = np.array([85, 90, 100, 60, 90, 75, 90])

is_ninety = (scores == 90)
print("90점인 점수 Boolean 배열:", is_ninety)
idx_ninety = np.where(scores == 90)
print(list(idx_ninety[0]))

print("90점인 학생 수:", (scores == 90).sum())
print("80점 이상인 학생 수:", (scores >= 80).sum())

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화
# Z-Score 표준화
np_arr_normalized = (np_arr - np.mean(np_arr)) / np.std(np_arr)
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0
# Min-Max 정규화
np_arr_final = ((np_arr - np.min(np_arr)) / (np.max(np_arr) - np.min(np_arr))).astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)