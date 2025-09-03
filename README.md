


## Building c++
So, I've developed it all on Windows, so it is buildable on Windows
To support linux, update CMakeLists.txt (write a new one :=^)

```ps
mkdir build
```

```ps
cmake -G "Visual Studio 17 2022" -A x64 ../cpp
```
<details>
  <summary>Output</summary>
  -- Selecting Windows SDK version 10.0.22621.0 to target Windows 10.0.26100.
-- The CXX compiler identification is MSVC 19.37.32825.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The C compiler identification is MSVC 19.37.32825.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Configuring done (4.4s)
-- Generating done (0.1s)
-- Build files have been written to: C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/build
</details>

```ps
cmake --build . --config Release -- /m
```
<details>
  <summary>Output</summary>
  MSBuild version 17.7.2+d6990bcfa for .NET Framework

  1>Checking Build System
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/ext/googletest-1.17
  .0/googlemock/CMakeLists.txt
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/ext/googletest-1.17
  .0/googletest/CMakeLists.txt
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/CMakeLists.txt
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/ext/googletest-1.17
  .0/googlemock/CMakeLists.txt
  gtest-all.cc
  utils.cpp
  gtest-all.cc
  gtest-all.cc
  gmock-all.cc
  utils.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\Release\utils.lib
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/CMakeLists.txt
  main.cpp
  gmock-all.cc
  gmock_main.cc
  gtest.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\lib\Release\gtest.lib
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/ext/googletest-1.17
  .0/googletest/CMakeLists.txt
  gtest_main.cc
  Generating Code...
  Generating Code...
  gtest_main.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\lib\Release\gtes
  t_main.lib
  CifarClassify.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\Release\Cifar
  Classify.exe
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/CMakeLists.txt
  test.cpp
  gmock.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\lib\Release\gmock.lib
  gmock_main.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\lib\Release\gmoc
  k_main.lib
  Tests.vcxproj -> C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\build\Release\Tests.exe
  Building Custom Rule C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/cpp/CMakeLists.txt
</details>

```ps
ctest --output-on-failure
```
<details>
  <summary>Output</summary>
  Test project C:/Users/Oleksii/Documents/OnlineEducation/ComputerVision/cifar_classify/build
    Start 1: TestPreprocessing.Normalize01
1/5 Test #1: TestPreprocessing.Normalize01 ..........   Passed    0.02 sec
    Start 2: TestPreprocessing.ResizeTo1x3x32x32
2/5 Test #2: TestPreprocessing.ResizeTo1x3x32x32 ....   Passed    0.02 sec
    Start 3: TestInference.ModelInitializes
3/5 Test #3: TestInference.ModelInitializes .........   Passed    0.08 sec
    Start 4: TestInference.ProbabilitiesAddsUpTo1
4/5 Test #4: TestInference.ProbabilitiesAddsUpTo1 ...   Passed    0.07 sec
    Start 5: TestInference.ExpectMeaningfulLabels
5/5 Test #5: TestInference.ExpectMeaningfulLabels ...   Passed    0.07 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   0.29 sec
</details>

```ps
.\Release\CifarClassify.exe --model ..\models\model.onnx --input ..\samples\cat.jpg
```
<details>
  <summary>Output</summary>
  class: cat, probability: 0.875078
</details>

```ps
.\Release\CifarClassify.exe --model ..\models\model.onnx --input ..\samples\automobile.jpg
```
<details>
  <summary>Output</summary>
  class: automobile, probability: 0.986446
</details>