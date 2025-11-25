#include "opencl.h"
#include "xgemm.h"
#include "xgemm_direct.cl"
#include "xgemm_mec.cl"

// #define TEST 1
#ifdef GPU

cl_program *opencl_im2col_mec_kernels_program;
cl_kernel *opencl_im2col_mec_kernel;

cl_program *opencl_gemm_mec_kernel_program;
cl_kernel *opencl_gemm_mec_kernel;

cl_program *xopencl_im2col_kernels_program;
cl_kernel *xopencl_im2col_gpu_kernel;

cl_program *xopencl_gemm_kernel_program;
cl_kernel *xopencl_gemm_kernel;

void xgemm_kernel_init(void)
{
    if (opencl_device_id_t == 0)
    {
        opencl_gemm_mec_kernel_program = (cl_program *)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_gemm_mec_kernel = (cl_kernel *)calloc(opencl_device_ct_t, sizeof(cl_kernel));

        xopencl_gemm_kernel_program = (cl_program *)calloc(opencl_device_ct_t, sizeof(cl_program));
        xopencl_gemm_kernel = (cl_kernel *)calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(xgemm_mec_kernel_source, strlen(xgemm_mec_kernel_source), &opencl_gemm_mec_kernel_program[opencl_device_id_t]);
    opencl_create_kernel(&opencl_gemm_mec_kernel_program[opencl_device_id_t], "xgemm_mec_kernel", &opencl_gemm_mec_kernel[opencl_device_id_t]);

    opencl_load_buffer(xgemm_direct_kernel_source, strlen(xgemm_direct_kernel_source), &xopencl_gemm_kernel_program[opencl_device_id_t]);
    opencl_create_kernel(&xopencl_gemm_kernel_program[opencl_device_id_t], "xgemm_kernel", &xopencl_gemm_kernel[opencl_device_id_t]);
}

void xgemm_kernel_release(void)
{
    clReleaseKernel(opencl_gemm_mec_kernel[opencl_device_id_t]);
    opencl_gemm_mec_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_gemm_mec_kernel_program[opencl_device_id_t]);
    opencl_gemm_mec_kernel_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t - 1)
    {
        free(opencl_gemm_mec_kernel);
    }

    clReleaseKernel(xopencl_gemm_kernel[opencl_device_id_t]);
    xopencl_gemm_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(xopencl_gemm_kernel_program[opencl_device_id_t]);
    xopencl_gemm_kernel_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t - 1)
    {
        free(xopencl_gemm_kernel);
    }
}

void xim2col_kernel_init(void)
{
    if (opencl_device_id_t == 0)
    {
        opencl_im2col_mec_kernels_program = (cl_program *)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_im2col_mec_kernel = (cl_kernel *)calloc(opencl_device_ct_t, sizeof(cl_kernel));

        xopencl_im2col_kernels_program = (cl_program *)calloc(opencl_device_ct_t, sizeof(cl_program));
        xopencl_im2col_gpu_kernel = (cl_kernel *)calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }
    opencl_load_buffer(xim2col_mec_kernel_source, strlen(xim2col_mec_kernel_source), &opencl_im2col_mec_kernels_program[opencl_device_id_t]);

    opencl_create_kernel(&opencl_im2col_mec_kernels_program[opencl_device_id_t], "xim2col_mec_kernel_nchw_gpu", &opencl_im2col_mec_kernel[opencl_device_id_t]);

    opencl_load_buffer(xim2col_kernel_source, strlen(xim2col_kernel_source), &xopencl_im2col_kernels_program[opencl_device_id_t]);
    opencl_create_kernel(&xopencl_im2col_kernels_program[opencl_device_id_t], "xim2col_gpu_kernel", &xopencl_im2col_gpu_kernel[opencl_device_id_t]);
}

void xim2col_kernel_release(void)
{
    clReleaseKernel(opencl_im2col_mec_kernel[opencl_device_id_t]);
    opencl_im2col_mec_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_im2col_mec_kernels_program[opencl_device_id_t]);
    opencl_im2col_mec_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t - 1)
    {
        free(opencl_im2col_mec_kernels_program);
        free(opencl_im2col_mec_kernel);
    }

    clReleaseKernel(xopencl_im2col_gpu_kernel[opencl_device_id_t]);
    xopencl_im2col_gpu_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(xopencl_im2col_kernels_program[opencl_device_id_t]);
    xopencl_im2col_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t - 1)
    {
        free(xopencl_im2col_kernels_program);
        free(xopencl_im2col_gpu_kernel);
    }
}

int CeilDiv(const int x, const int y)
{
    return 1 + ((x - 1) / y);
}
int Ceil(const int x, const int y)
{
    return CeilDiv(x, y) * y;
}

void ProcessArguments(const int layout, const int a_transpose, const int b_transpose,
                      int *a_do_transpose, int *b_do_transpose, int *c_do_transpose,
                      int *a_conjugate, int *b_conjugate)
{
    // layout : 1 = CblasColMajor, 0 = CblasRowMajor
    // a_transpose : 1 = trans, 0 = notrans
    const int a_rotated = (layout == 1 && a_transpose != 0) ||
                          (layout == 0 && a_transpose == 0);
    const int b_rotated = (layout == 1 && b_transpose != 0) ||
                          (layout == 0 && b_transpose == 0);
    const int c_rotated = (layout == 0);
    *a_do_transpose = a_rotated != 0;
    *b_do_transpose = b_rotated != 1;
    *c_do_transpose = c_rotated != 0;

    // In case of complex data-types, the transpose can also become a conjugate transpose
    *a_conjugate = 0; //(a_transpose == CblasConjTrans);
    *b_conjugate = 0; //(b_transpose == CblasConjTrans);
}

// void xim2col_mec_gpu(cl_mem_ext im, int offset,
//                      int channels, int height, int width,
//                      int ksize, int stride, int pad, cl_mem_ext data_col)
// {

//     int batch = 1;
//     // int height_col = (height + 2 * pad - ksize) / stride + 1;
//     // int width_col = (width + 2 * pad - ksize) / stride + 1;
//     // int h = stride > ksize ? width_col * ksize : width_col * stride + ksize - stride;
//     // int strideH = stride > ksize ? stride : ksize;

//     // dim3 dimGridG1;
//     // dimGridG1 = dim3_create(batch, channels, 1);
//     // dim2 dimGrid;
//     // dimGrid = dim2_create(channels, 1);
//     // // printf(">> xim2col_mec_gpu : channels:%d\n",channels);

//     // opencl_kernel(
//     //     opencl_im2col_mec_kernel[opencl_device_id_t], dimGrid, 24,
//     //     &im, sizeof(cl_mem),
//     //     &height, sizeof(cl_int),
//     //     &width, sizeof(cl_int),
//     //     &ksize, sizeof(cl_int),
//     //     &pad, sizeof(cl_int),
//     //     &stride, sizeof(cl_int),
//     //     &height_col, sizeof(cl_int),
//     //     &width_col, sizeof(cl_int),
//     //     &data_col, sizeof(cl_mem),
//     //     &h, sizeof(cl_int),
//     //     &batch, sizeof(cl_int),
//     //     &channels, sizeof(cl_int));

//     int height_col = (height + 2 * pad - ksize) / stride + 1;
//     int width_col = (width + 2 * pad - ksize) / stride + 1;
//     // int num_kernels = batch * channels * height_col * width_col;
//     int mec_h = stride > ksize ? width_col * ksize : width_col * stride + ksize - stride;

//     // printf(">> xim2col_mec_gpu : channels:%d,height_col:%d,height:%d,stride:%d,pad:%d\n", channels, height_col, height, stride,pad);
//     dim2 dimGrid;
//     dimGrid = dim2_create(channels, width_col);

//     opencl_kernel(
//         opencl_im2col_mec_kernel[opencl_device_id_t], dimGrid, 24,
//         &im, sizeof(cl_mem),
//         &height, sizeof(cl_int),
//         &width, sizeof(cl_int),
//         &ksize, sizeof(cl_int),
//         &pad, sizeof(cl_int),
//         &stride, sizeof(cl_int),
//         &height_col, sizeof(cl_int),
//         &width_col, sizeof(cl_int),
//         &data_col, sizeof(cl_mem),
//         &mec_h, sizeof(cl_int),
//         &channels, sizeof(cl_int),
//         &batch, sizeof(cl_int));
//     // float *img_list = (float *)malloc(sizeof(float) * batch * channels * height * width);
//     // float *im2col_list = (float *)malloc(sizeof(float) * batch * channels * mec_h * ksize * height_col);
//     // opencl_pull_array(im, img_list, batch * channels * height * width);
//     // int b = 0; // get_global_id(0);

//     // for (int c = 0; c < channels; c++)
//     // {
//     //     for (int ow = 0; ow < width_col; ow++)
//     //     {
//     //         for (int h = 0; h < mec_h; h++)
//     //         {
//     //             for (int k = 0; k < ksize; k++)
//     //             {
//     //                 im2col_list[b * width_col * mec_h * ksize * channels + c * width_col * mec_h * ksize + ow + h * ksize * width_col + k * width_col] =
//     //                     img_list[b * channels * height * width + c * height * width + h * width + (ow * stride + k)];
//     //             }
//     //         }
//     //     }
//     // }
//     // opencl_push_array(data_col, im2col_list, batch * channels * mec_h * ksize * height_col);
//     // free(img_list);
//     // free(im2col_list);
// }


// ******************fx promax版**********************
// ******************fx promax版**********************
// ******************fx promax版**********************
void xim2col_mec_gpu(cl_mem_ext im, int offset,
    int channels, int height, int width,
    int ksize, int stride, int pad, int batch, cl_mem_ext data_col)
{

// int batch = 1;
// int height_col = (height + 2 * pad - ksize) / stride + 1;
// int width_col = (width + 2 * pad - ksize) / stride + 1;
// int h = stride > ksize ? width_col * ksize : width_col * stride + ksize - stride;
// int strideH = stride > ksize ? stride : ksize;

// dim3 dimGridG1;
// dimGridG1 = dim3_create(batch, channels, 1);
// dim2 dimGrid;
// dimGrid = dim2_create(channels, 1);
// // printf(">> xim2col_mec_gpu : channels:%d\n",channels);

// opencl_kernel(
//     opencl_im2col_mec_kernel[opencl_device_id_t], dimGrid, 24,
//     &im, sizeof(cl_mem),
//     &height, sizeof(cl_int),
//     &width, sizeof(cl_int),
//     &ksize, sizeof(cl_int),
//     &pad, sizeof(cl_int),
//     &stride, sizeof(cl_int),
//     &height_col, sizeof(cl_int),
//     &width_col, sizeof(cl_int),
//     &data_col, sizeof(cl_mem),
//     &h, sizeof(cl_int),
//     &batch, sizeof(cl_int),
//     &channels, sizeof(cl_int));

int height_col = (height + 2 * pad - ksize) / stride + 1;
int width_col = (width + 2 * pad - ksize) / stride + 1;
// int num_kernels = batch * channels * height_col * width_col;
int mec_h = stride > ksize ? width_col * ksize : width_col * stride + ksize - stride;

// printf(">> xim2col_mec_gpu : channels:%d,height_col:%d,height:%d,stride:%d,pad:%d\n", channels, height_col, height, stride,pad);
dim2 dimGrid;
dimGrid = dim2_create(channels, width_col);


opencl_kernel(
opencl_im2col_mec_kernel[opencl_device_id_t], dimGrid, 24,
&im, sizeof(cl_mem),
&height, sizeof(cl_int),
&width, sizeof(cl_int),
&ksize, sizeof(cl_int),
&pad, sizeof(cl_int),
&stride, sizeof(cl_int),
&height_col, sizeof(cl_int),
&width_col, sizeof(cl_int),
&data_col, sizeof(cl_mem),
&mec_h, sizeof(cl_int),
&channels, sizeof(cl_int),
&batch, sizeof(cl_int));
// float *img_list = (float *)malloc(sizeof(float) * batch * channels * height * width);
// float *im2col_list = (float *)malloc(sizeof(float) * batch * channels * mec_h * ksize * height_col);
// opencl_pull_array(im, img_list, batch * channels * height * width);
// int b = 0; // get_global_id(0);

// for (int c = 0; c < channels; c++)
// {
//     for (int ow = 0; ow < width_col; ow++)
//     {
//         for (int h = 0; h < mec_h; h++)
//         {
//             for (int k = 0; k < ksize; k++)
//             {
//                 im2col_list[b * width_col * mec_h * ksize * channels + c * width_col * mec_h * ksize + ow + h * ksize * width_col + k * width_col] =
//                     img_list[b * channels * height * width + c * height * width + h * width + (ow * stride + k)];
//             }
//         }
//     }
// }
// opencl_push_array(data_col, im2col_list, batch * channels * mec_h * ksize * height_col);
// free(img_list);
// free(im2col_list);
}

// void xgemm_mec_gpu(int TA, int TB,
//                    cl_mem_ext kerne_mec, cl_mem_ext img2col, cl_mem_ext output_mec,
//                    int img_size, int kernel_size, int channels, int stride,
//                    int out_channels, int batch, int pad)
// {
//     // printf(">> xgemm_mec_gpu\n");
//     // printf("gemm gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
//     const int output_size = (img_size + 2 * pad - kernel_size) / stride + 1;
//     const int h = stride > kernel_size ? output_size * kernel_size : output_size * stride + kernel_size - stride;

//     const int M = out_channels;
//     const int N = output_size * output_size;

//     const int K = kernel_size * kernel_size * channels;
//     const int b_ld = output_size;

//     const int a_ld = kernel_size * kernel_size * channels;

//     const int c_ld = output_size * output_size;
//     int a_offset = 0;
//     int b_offset = 0;
//     int c_offset = 0;
//     float ALPHA = 1;
//     float BETA = 0;

//     int WGD = 32;
//     int MDIMCD = 8;
//     int NDIMCD = 8;
//     const int NWID = (WGD / NDIMCD);

//     const int m_ceiled = Ceil(M, WGD);
//     const int n_ceiled = Ceil(N, WGD);

//     dim3 dimGridG1;
//     dimGridG1 = dim3_create(m_ceiled * MDIMCD / WGD, n_ceiled * NDIMCD / WGD, batch);
//     dim3 dimGridL1;
//     dimGridL1 = dim3_create(MDIMCD, NDIMCD, 1);

//     int layer = 0;

//     int a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
//     ProcessArguments(layer, TA, TB, &a_do_transpose, &b_do_transpose, &c_do_transpose, &a_conjugate, &b_conjugate);

//     int test = 0;
//     opencl_kernel_local3(
//         opencl_gemm_mec_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 56,
//         &M, sizeof(cl_int),
//         &N, sizeof(cl_int),
//         &K, sizeof(cl_int),
//         &ALPHA, sizeof(cl_float),
//         &BETA, sizeof(cl_float),
//         &kerne_mec.mem, sizeof(cl_mem),
//         &a_offset, sizeof(cl_int),
//         &a_ld, sizeof(cl_int),
//         &img2col.mem, sizeof(cl_mem),
//         &b_offset, sizeof(cl_int),
//         &b_ld, sizeof(cl_int),
//         &output_mec.mem, sizeof(cl_mem),
//         &c_offset, sizeof(cl_int),
//         &c_ld, sizeof(cl_int),
//         &a_do_transpose, sizeof(cl_int),
//         &b_do_transpose, sizeof(cl_int),
//         &c_do_transpose, sizeof(cl_int),
//         &a_conjugate, sizeof(cl_int),
//         &b_conjugate, sizeof(cl_int),
//         // add
//         &img_size, sizeof(cl_int),
//         &kernel_size, sizeof(cl_int),
//         &channels, sizeof(cl_int),
//         &stride, sizeof(cl_int),
//         &out_channels, sizeof(cl_int),
//         &batch, sizeof(cl_int),
//         &h, sizeof(cl_int),
//         &output_size, sizeof(cl_int),
//         &test, sizeof(cl_int));
//     // printf(">> xgemm_mec_gpu end\n");
// }


//******************fx promax版**********************
//******************fx promax版**********************
//******************fx promax版**********************
void xgemm_mec_gpu(int TA, int TB,
    cl_mem_ext kerne_mec, cl_mem_ext img2col, cl_mem_ext output_mec,
    int img_size, int kernel_size, int channels, int stride,
    int out_channels, int batch, int groups, int pad)
{
// printf(">> xgemm_mec_gpu\n");
// printf("gemm gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
const int output_size = (img_size + 2 * pad - kernel_size) / stride + 1;
const int h = stride > kernel_size ? output_size * kernel_size : output_size * stride + kernel_size - stride;

const int M = out_channels;
const int N = output_size * output_size;

const int K = kernel_size * kernel_size * channels;
const int b_ld = output_size;

const int a_ld = kernel_size * kernel_size * channels;

const int c_ld = output_size * output_size;
int a_offset = 0;
int b_offset = 0;
int c_offset = 0;
float ALPHA = 1;
float BETA = 0;

int WGD = 32;
int MDIMCD = 8;
int NDIMCD = 8;
const int NWID = (WGD / NDIMCD);

const int m_ceiled = Ceil(M, WGD);
const int n_ceiled = Ceil(N, WGD);

dim3 dimGridG1;
dimGridG1 = dim3_create(m_ceiled * MDIMCD / WGD, n_ceiled * NDIMCD / WGD, batch * groups);
dim3 dimGridL1;
dimGridL1 = dim3_create(MDIMCD, NDIMCD, 1);

int layer = 0;

int a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
ProcessArguments(layer, TA, TB, &a_do_transpose, &b_do_transpose, &c_do_transpose, &a_conjugate, &b_conjugate);

int test = 0;
opencl_kernel_local3(
opencl_gemm_mec_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 58,
&M, sizeof(cl_int),
&N, sizeof(cl_int),
&K, sizeof(cl_int),
&ALPHA, sizeof(cl_float),
&BETA, sizeof(cl_float),
&kerne_mec.mem, sizeof(cl_mem),
&a_offset, sizeof(cl_int),
&a_ld, sizeof(cl_int),
&img2col.mem, sizeof(cl_mem),
&b_offset, sizeof(cl_int),
&b_ld, sizeof(cl_int),
&output_mec.mem, sizeof(cl_mem),
&c_offset, sizeof(cl_int),
&c_ld, sizeof(cl_int),
&a_do_transpose, sizeof(cl_int),
&b_do_transpose, sizeof(cl_int),
&c_do_transpose, sizeof(cl_int),
&a_conjugate, sizeof(cl_int),
&b_conjugate, sizeof(cl_int),
// add
&img_size, sizeof(cl_int),
&kernel_size, sizeof(cl_int),
&channels, sizeof(cl_int),
&stride, sizeof(cl_int),
&out_channels, sizeof(cl_int),
&batch, sizeof(cl_int),
&h, sizeof(cl_int),
&output_size, sizeof(cl_int),
&groups, sizeof(cl_int),
&test, sizeof(cl_int));
// printf(">> xgemm_mec_gpu end\n");
}

void xim2col_gpu(cl_mem im, int offset,
                 int channels, int height, int width,
                 int ksize, int stride, int pad, cl_mem data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    dim2 dimGrid;
    dimGrid = dim2_create(num_kernels, 1);
    //  printf(">> xim2col_gpu : channels:%d,height_col:%d,height:%d,stride:%d\n", channels, height_col, height, stride);
    // printf("num_kernels:%d\n", num_kernels);

    opencl_kernel(
        xopencl_im2col_gpu_kernel[opencl_device_id_t], dimGrid, 20,
        &num_kernels, sizeof(cl_int),
        &im, sizeof(cl_mem),
        &height, sizeof(cl_int),
        &width, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &height_col, sizeof(cl_int),
        &width_col, sizeof(cl_int),
        &data_col, sizeof(cl_mem));
}

void xgemm_gpu(int TA, int TB, int M, int N, int K,
               float ALPHA,
               cl_mem_ext kernel, int a_offset, int a_ld,
               cl_mem_ext img2col, int b_offset, int b_ld,
               float BETA,
               cl_mem_ext output, int c_offset, int c_ld,int Batch)
{

    // printf("gemm gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    // const int output_size = (img_size + 2 * pad - kernel_size) / stride + 1;
    // const int h = stride > kernel_size ? output_size * kernel_size : output_size * stride + kernel_size - stride;

    // const int M = out_channels;
    // const int N = output_size * output_size;
    // const int K = kernel_size * kernel_size * channels;
    // const int a_ld = kernel_size * kernel_size * channels;
    // const int b_ld = output_size * output_size;
    // const int c_ld = output_size * output_size;

    // int a_offset = 0;
    // int b_offset = 0;
    // int c_offset = 0;
    // float ALPHA = 1;
    // float BETA = 0;
    int batch = 1;
    // int batch=2;
    int WGD = 32;
    int MDIMCD = 8;
    int NDIMCD = 8;
    const int NWID = (WGD / NDIMCD);

    const int m_ceiled = Ceil(M, WGD);
    const int n_ceiled = Ceil(N, WGD);

    dim3 dimGridG1;
    dimGridG1 = dim3_create(m_ceiled * MDIMCD / WGD, n_ceiled * NDIMCD / WGD, batch);
    dim3 dimGridL1;
    dimGridL1 = dim3_create(MDIMCD, NDIMCD, 1);

    int a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    ProcessArguments(0, TA, TB, &a_do_transpose, &b_do_transpose, &c_do_transpose, &a_conjugate, &b_conjugate);

    int test = 0;
    opencl_kernel_local3(
        xopencl_gemm_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 40,
        &M, sizeof(cl_int),
        &N, sizeof(cl_int),
        &K, sizeof(cl_int),
        &ALPHA, sizeof(cl_float),
        &BETA, sizeof(cl_float),
        &kernel.mem, sizeof(cl_mem),
        &a_offset, sizeof(cl_int),
        &a_ld, sizeof(cl_int),
        &img2col.mem, sizeof(cl_mem),
        &b_offset, sizeof(cl_int),
        &b_ld, sizeof(cl_int),
        &output.mem, sizeof(cl_mem),
        &c_offset, sizeof(cl_int),
        &c_ld, sizeof(cl_int),
        &a_do_transpose, sizeof(cl_int),
        &b_do_transpose, sizeof(cl_int),
        &c_do_transpose, sizeof(cl_int),
        &a_conjugate, sizeof(cl_int),
        &b_conjugate, sizeof(cl_int),

        // &kernel_size, sizeof(cl_int),
        // &channels, sizeof(cl_int),
        // &output_size, sizeof(cl_int),
        // &out_channels, sizeof(cl_int),
        &test, sizeof(cl_int));
}
#endif
