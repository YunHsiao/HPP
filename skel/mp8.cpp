
#include <wb.h>
#include <amp.h>

using namespace concurrency;

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  //@@ Insert C++AMP code here
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  array_view<float, 1> viewA(inputLength, hostInput1), viewB(inputLength, hostInput2);
  array_view<float, 1> viewC(inputLength, hostOutput);
  viewC.discard_data();
  parallel_for_each(viewC.extent, [=](index<1> i) restrict(amp) {
    viewC[i] = viewA[i] + viewB[i];
  });
  viewC.synchronize();
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
