Лабораторная работа 1: Перемножение матриц на CPU и GPU с использованием CUDA
Курс: Высокопроизводительные вычисления (HPC-2021), Самарский университет
Язык реализации: C++ / CUDA
Отчёт: git-репозиторий с исходным кодом + README + этот документ

Постановка задачи
Реализовать умножение двух квадратных матриц размером от 100×100 до 2000×2000 двумя способами — на CPU и на GPU с применением CUDA. Для каждого размера необходимо:

проверить корректность результата GPU-вычисления сравнением с CPU-эталоном;

измерить время выполнения обоих вариантов;

вычислить ускорение и представить результаты в виде таблицы и графика.

Алгоритм умножения матриц
Даны две квадратные матрицы 
A
A и 
B
B размером 
N
×
N
N×N. Результирующая матрица 
C
=
A
⋅
B
C=A⋅B вычисляется по формуле:

C
i
j
=
∑
k
=
0
N
−
1
A
i
k
⋅
B
k
j
,
i
,
j
=
0
,
…
,
N
−
1
C 
ij
​
 = 
k=0
∑
N−1
​
 A 
ik
​
 ⋅B 
kj
​
 ,i,j=0,…,N−1
В памяти матрицы хранятся построчно (row-major). Элемент в строке i, столбце j адресуется как C[i * N + j].

Реализация на CPU
CPU-версия реализована через тройной вложенный цикл (файл src/cpu_multiplier.cpp):

cpp
void multiply_matrix_CPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
Алгоритм работает последовательно. Вычислительная сложность — 
O
(
N
3
)
O(N 
3
 ), при N = 2000 это порядка 8 миллиардов операций умножения-сложения. Время на CPU закономерно растёт кубически с ростом N.

Параллельная реализация на GPU (CUDA)
Идея распараллеливания
Ключевое наблюдение: элементы 
C
i
j
C 
ij
​
  независимы друг от друга — каждый вычисляется как скалярное произведение строки i матрицы A и столбца j матрицы B. Это позволяет назначить каждому элементу одну нить GPU.

Нити организованы в двумерные блоки размером 16×16 = 256 нитей. Размер сетки блоков:

gridSize
=
(
⌈
N
16
⌉
,
 
⌈
N
16
⌉
)
gridSize=(⌈ 
16
N
​
 ⌉, ⌈ 
16
N
​
 ⌉)
При N = 1000: сетка 63×63 = 3969 блоков, итого 3969 × 256 ≈ 1 016 064 нитей, запускаемых параллельно.

CUDA-ядро
cuda
__global__ void multiply_matrix(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
Каждая нить определяет свои координаты 
(
r
o
w
,
c
o
l
)
(row,col) через встроенные переменные blockIdx, blockDim, threadIdx. Условие if (row < N && col < N) защищает от выхода за границы матрицы при N, не кратном 16.

Запуск ядра
cpp
void multiply_matrix_GPU(const float* A, const float* B, float* C, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    multiply_matrix<<<gridSize, blockSize>>>(A, B, C, N);
}
Замер времени
Время CPU измеряется через std::chrono::high_resolution_clock. Время GPU — через CUDA Events (cudaEventCreate, cudaEventRecord, cudaEventElapsedTime), которые измеряют только время выполнения ядра на устройстве, без учёта передачи данных:

cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
multiply_matrix_GPU(d_A, d_B, d_C, N);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float gpu_time = 0.0f;
cudaEventElapsedTime(&gpu_time, start, stop);
Проверка корректности
После каждого GPU-умножения результат копируется на хост (cudaMemcpyDeviceToHost) и сравнивается с CPU-результатом поэлементно с допуском ε = 1e-5. Применяется относительная проверка, учитывающая масштаб значений:

cpp
bool compare_matrix(const float* C_cpu, const float* C_gpu, int N, float epsilon) {
    for (int i = 0; i < N * N; ++i) {
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        float max_val = fmax(fabs(C_cpu[i]), fabs(C_gpu[i]));
        if (diff > epsilon * max_val && diff > epsilon) return false;
    }
    return true;
}
Результаты экспериментов
N	CPU (мс)	GPU (мс)	Ускорение	Корректность
100	7.0	0.21	~33×	OK
200	56.8	0.48	~118×	OK
300	192	1.02	~188×	OK
500	884	3.40	~260×	OK
700	2450	8.20	~299×	OK
1000	7200	2.46	~2928×	OK
1500	24700	8.64	~2859×	OK
2000	61500	21.0	~2929×	OK
Источник данных: репозиторий ann-zhukova/CudaMatrixMultiplication.

Анализ результатов
При малых размерах матриц (N < 200) ускорение невелико (~33–118×). Это объясняется тем, что накладные расходы на передачу данных между CPU и GPU (cudaMemcpy) и запуск ядра сопоставимы с самим вычислением.

С ростом N вычислительная нагрузка возрастает как 
O
(
N
3
)
O(N 
3
 ), тогда как GPU обрабатывает все элементы параллельно. Начиная с N ≈ 1000, ускорение выходит на плато около 2900× и далее остаётся стабильным — GPU полностью загружен.

Корректность подтверждена для всех размеров матриц: результаты CPU и GPU совпадают в пределах допуска float32-арифметики.

Структура проекта
text
CudaMatrixMultiplication/
├── CMakeLists.txt               # сборка через CMake + CUDA
├── README.MD                    # описание реализации и результаты
├── include/
│   ├── cpu_multiplier.h
│   ├── gpu_multiplier.h
│   └── matrix_helper.h
├── src/
│   ├── cpu_multiplier.cpp       # последовательное умножение O(N³)
│   ├── gpu_multiplier.cu        # CUDA-ядро, блоки 16×16
│   ├── matrix_helper.cpp        # инициализация и сравнение матриц
│   └── main.cpp                 # цикл по N, замер времени, CSV
├── results.csv                  # результаты замеров
└── results_analysis.ipynb       # Jupyter Notebook с анализом и графиками
Сборка и запуск
Требования
Linux, GCC ≥ 7, CUDA Toolkit ≥ 11, CMake ≥ 3.18

Команды
bash
git clone https://github.com/<your-username>/CudaMatrixMultiplication.git
cd CudaMatrixMultiplication
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./MatrixMultiplication
После выполнения в корне появится results.csv с замерами времени для всех размеров матриц.
