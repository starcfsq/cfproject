#ifdef GPU

#include "darknet.h"

#include <string.h>

#include "activations.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "opencl.h"
#include "convolutional_kernels.cl"
#include "layer.h"
#include "xgemm.h"

cl_program* opencl_convolutional_kernels_program;
cl_kernel* opencl_binarize_kernel;
cl_kernel* opencl_binarize_input_kernel;
cl_kernel* opencl_binarize_weights_kernel;
cl_kernel* opencl_smooth_kernel;

void convolutional_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_convolutional_kernels_program = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_binarize_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_binarize_input_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_binarize_weights_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_smooth_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(convolutional_kernel_source, strlen(convolutional_kernel_source), &opencl_convolutional_kernels_program[opencl_device_id_t]);

    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_kernel", &opencl_binarize_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_input_kernel", &opencl_binarize_input_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_weights_kernel", &opencl_binarize_weights_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "smooth_kernel", &opencl_smooth_kernel[opencl_device_id_t]);

}

void convolutional_kernel_release(void)
{
    clReleaseKernel(opencl_binarize_kernel[opencl_device_id_t]); opencl_binarize_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_binarize_input_kernel[opencl_device_id_t]); opencl_binarize_input_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_binarize_weights_kernel[opencl_device_id_t]); opencl_binarize_weights_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_smooth_kernel[opencl_device_id_t]); opencl_smooth_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_convolutional_kernels_program[opencl_device_id_t]); opencl_convolutional_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_convolutional_kernels_program);
        free(opencl_binarize_kernel);
        free(opencl_binarize_input_kernel);
        free(opencl_binarize_weights_kernel);
        free(opencl_smooth_kernel);
    }
}

void binarize_gpu(cl_mem x, int n, cl_mem binary)
{

    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_binarize_kernel[opencl_device_id_t], dimN, 6, &x, sizeof(cl_mem), &n, sizeof(cl_int), &binary, sizeof(cl_mem));
}


void binarize_input_gpu(cl_mem input, int n, int size, cl_mem binary)
{
    dim2 dimN;
    dimN = opencl_gridsize(size);

    opencl_kernel(opencl_binarize_input_kernel[opencl_device_id_t], dimN, 8, &input, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));
}


void binarize_weights_gpu(cl_mem weights, int n, int size, cl_mem binary)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_binarize_weights_kernel[opencl_device_id_t], dimN, 8, &weights, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));
}

void swap_binary_gpu(convolutional_layer *l)
{
    cl_mem_ext swap_gpu = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap_gpu;
}

// void forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1);

//     if (l.binary) {
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if (l.xnor) {
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c * l.h * l.w * l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }
//     int i;
//     int m = l.n / l.groups;  // Number of output channels per group
//     int k = l.size * l.size * l.c / l.groups;  // Input channels per group
//     int n = l.out_w * l.out_h;  // Output height * width

//     // Process each group
    
//     for (int j = 0; j < l.groups; ++j) {
//         for(i = 0; i < l.batch; ++i){
//         cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
//         cl_mem_ext b = net.workspace_gpu;
//         cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
//         cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
//         if (l.size == 1){
//                 b = im;
//             } else {
//                 // im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 im2col_gpu_offset(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem,(i*l.groups + j)*k*n);
//             }

//         }

//         xgemm_gpu(0,0,m,n,k,1,l.weights_gpu,j*l.nweights/l.groups,k,(l.size == 1)?net.input_gpu:net.workspace_gpu,j * k * n,n,1,l.output_gpu,j * (m * n),n,1);
//         // gemm_offset_gpu_strided_batched(0, 0, m, n, k,
//         //                         1,
//         //                         l.weights_gpu, j * l.nweights / l.groups, k, 0,
//         //                         (l.size == 1)?net.input_gpu:net.workspace_gpu, j * k * n, n, l.groups * k * n,
//         //                         1,
//         //                         l.output_gpu, j * (m * n), n, (m * n * l.groups),
//         //                         l.batch);
//     }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w * l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);

//     if (l.binary || l.xnor) {
//         swap_binary_gpu(&l);
//     }
// }
// forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     // ================ 核心修改部分 ================
//     int m = l.n;             // 合并所有groups的输出通道数
//     int k = l.size*l.size*l.c; // 合并所有groups的输入通道数
//     int n = l.out_w*l.out_h * l.batch; // 扩展批次维度到矩阵列数

//     cl_mem_ext a = l.weights_gpu;
//     cl_mem_ext b = net.workspace_gpu;
//     cl_mem_ext c = l.output_gpu;
//     cl_mem_ext im = net.input_gpu;

//     if (l.size == 1) {
//         // Case 1: 1x1卷积直接使用原输入矩阵
//         // 输入矩阵维度: [c, h*w*batch] (每个样本的h*w在内存中连续)
//         xgemm_gpu(0, 0, m, n, k, 1,
//                   a, 0, k,          // weights [n, c] (合并所有groups)
//                   im, 0, n, // input [c, h*w*batch]
//                   1, c, 0, n,1);       // output [n, h*w*batch]
//     } else {
//         // Case 2: 非1x1卷积需要合并所有批次的im2col
//         // 生成整个batch的im2col矩阵 [k, h_out*w_out*batch]
//         // xim2col_gpu_batch(im.mem, l.c, l.h, l.w, l.size, l.stride, l.pad,
//         //                  b.mem, l.batch); // 需自定义批量im2col函数
//         im2col_gpu(im.mem, 0,l.c, l.h, l.w, l.size, l.stride, l.pad,
//                             b.mem);
//         // 执行大矩阵乘法 [m, n] = [n, k] * [k, n]
//         xgemm_gpu(0, 0, m, n, k, 1,
//                   a, 0, k,          // weights [n, k] (合并所有groups)
//                   b, 0, n,         // im2col [k, h_out*w_out*batch]
//                   1, c, 0, n,1);      // output [n, h_out*w_out*batch]
//     }
//     // ================ 修改结束 ================

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }
// forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int j;
//     int m = l.n/l.groups;
//     int k = l.size*l.size*l.c/l.groups;
//     int n = l.out_w*l.out_h;
//     int batch = l.batch;

//     for(j = 0; j < l.groups; ++j){
//         cl_mem_ext a = l.weights_gpu;
//         cl_mem_ext b = net.workspace_gpu;
//         cl_mem_ext c = l.output_gpu;
//         cl_mem_ext im = net.input_gpu;

//         if (l.size == 1) {
//             // 处理1x1卷积，直接使用输入数据，无需im2col
//             // 输入的每个样本组j的偏移量间隔为groups*(c/groups)*h*w，即每个样本的总大小
//             xgemm_gpu(0, 0, m, n*batch, k, 1,
//                       a, j*l.nweights/l.groups, k,
//                       im, j*(l.c/l.groups)*l.h*l.w, l.groups*(l.c/l.groups)*l.h*l.w,
//                       1, c, j*m*n, m*n*l.groups,1);
//         } else {
//             // 生成整个批次的im2col矩阵
//             for(int i = 0; i < batch; ++i){
//                 int input_offset = (i*l.groups + j) * (l.c/l.groups)*l.h*l.w;
//                 int im2col_offset = i * k * n; // 每个样本的im2col结果占k*n元素
//                 im2col_gpu_offset(im.mem, input_offset, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad,
//                             b.mem,im2col_offset);
//             }
//             // 执行矩阵乘法，处理整个批次
//             xgemm_gpu(0, 0, m, n*batch, k, 1,
//                       a, j*l.nweights/l.groups, k,
//                       b, 0, n*batch,
//                       1, c, j*m*n, m*n*l.groups,1);
//         }
//     }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }
// void forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int i, j;
//     int m = l.n/l.groups;
//     int k = l.size*l.size*l.c/l.groups;
//     int n = l.out_w*l.out_h;
//     // double count_start_time=what_time_is_it_now();
//     for(i = 0; i < l.batch; ++i){
//         for(j = 0; j < l.groups; ++j){
//             cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
//             cl_mem_ext b = net.workspace_gpu;
//             cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
//             cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

//             if (l.size == 1){
//                 b = im;
//                 xgemm_gpu(0,0,m,n,k,1,a,0,k,b,i*l.c*l.h*l.w,n,1,c,(i*l.groups)*n*m,n,l.groups);
//                 // gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n);
//             } else {
//                 im2col_gpu(im.mem, (i*l.groups)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 // im2col_gpu_offset(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem,(i*l.groups + j)*k*n);
//                 // xim2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 // gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
//                 xgemm_gpu(0,0,m,n,k,1,a,0,k,b,0,n,1,c,(i*l.groups)*n*m,n,l.groups);
//             }
//         }
//     }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }
//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }




// void forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int i, j;
//     int m = l.n/l.groups;
//     int k = l.size*l.size*l.c/l.groups;
//     int n = l.out_w*l.out_h;
//     // double count_start_time=what_time_is_it_now();
//     for(i = 0; i < l.batch; ++i){
//         for(j = 0; j < l.groups; ++j){
//             cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
//             cl_mem_ext b = net.workspace_gpu;
//             cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
//             cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

//             if (l.size == 1){
//                 b = im;
//                 // xgemm_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n,l.groups);
//                 // gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n);
//             } else {
//                 im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 // im2col_gpu_offset(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem,(i*l.groups + j)*k*n);
//                 // xim2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 // gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
//                 // xgemm_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n,l.groups);
//             }
//             xgemm_gpu(0,0,m,n,k,1,a,0,k,b,0,n,1,c,i*n*m,n,l.groups);
//         }
//     }

//     //输出卷积层的矩阵运算时间
//     // clFinish(opencl_queues[opencl_device_id_t]);
//     // static int which_layer=0;
//     // double count_end_time=what_time_is_it_now();
//     // static double total_time= 0;
//     // total_time+=count_end_time-count_start_time;
//     // if(which_layer==17)
//     // {
//     //     printf("count_time : %f\n",total_time); 
//     // }
//     // which_layer++;


//     // static int which_layer=0;
//     // float *fc=(float*)clEnqueueMapBuffer(l.output_gpu.que,l.output_gpu.mem,CL_TRUE,CL_MAP_READ,0,sizeof(float)*n*m,0,NULL,NULL,NULL);
//     //验证输出结果
//     // printf("%dconvolutional:    ", which_layer);
//     // for(int i = 0 ; i < 2; i++){
//     //     printf("%f ",fc[i]);
        
//     // }
//     // which_layer++;
//     // printf("\n");
//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     //if(l.dot > 0) dot_error_gpu(l);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }

//***************************fx版本正确版*****************************
//***************************fx版本正确版*****************************
//***************************fx版本正确版*****************************
// void forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int i, j;
//     int m = l.n/l.groups;
//     int k = l.size*l.size*l.c/l.groups;
//     int n = l.out_w*l.out_h;
//     for(i = 0; i < l.batch; ++i){
//         for(j = 0; j < l.groups; ++j){
//             cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
//             cl_mem_ext b = net.workspace_gpu;
//             cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
//             cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

//             if (l.size == 1){
//                 b = im;
//                 xgemm_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n,l.groups);
//             } else {
//                 // im2col_gpu_offset(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem,(i*l.groups + j)*k*n);
//                 //im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
//                 //fangxin
//                 // xim2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 //xgemm_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n,l.groups);
 
//                 xim2col_mec_gpu(im,(i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//                 xgemm_mec_gpu(0,0,a,b,c,l.h,l.size, l.c/l.groups,l.stride,m,1,l.pad);

//             }
//         }
//    }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     //if(l.dot > 0) dot_error_gpu(l);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }

//********************将l.groups当成batch计算**********************
// forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int i, j;
//     int m = l.n / l.groups;
//     int k = l.size * l.size * l.c / l.groups;
//     int n = l.out_w * l.out_h;

    
//     // int m = l.n / l.groups;  // Number of output channels per group
//     // int k = l.size * l.size * l.c / l.groups;  // Input channels per group
//     // int n = l.out_w * l.out_h;  // Output height * width
//     for(i = 0; i < l.batch; ++i){
//         cl_mem_ext a = l.weights_gpu;
//         cl_mem_ext b = net.workspace_gpu;
//         cl_mem_ext c = l.output_gpu;
//         cl_mem_ext im = net.input_gpu;

//         if (l.size == 1) {
//             // ========= 1x1卷积的特殊处理 =========
//             // 输入矩阵布局: [c, h*w] -> 扩展为 [c, h*w*groups]
//             // 权重矩阵布局: [n, c] (已包含所有groups)
//             // gemm_offset_gpu(0,0,m,l.out_w*l.out_h*l.groups,k/l.groups,1,
//             //         a, 0, k, // weights [n, c]
//             //         im, i*l.c*l.h*l.w, l.c*l.h*l.w, // 输入偏移按batch计算
//             //         1,
//             //         c, i*m*l.out_w*l.out_h, m);
//             b = im;
//         } else {
//             // ========= 非1x1卷积处理 =========
//             // 步骤1: 为整个batch的当前样本生成所有groups的im2col
//             for(j = 0; j < l.groups; ++j){
//                 int im_offset = (i*l.groups + j)*l.c/l.groups*l.h*l.w;
//                 int col_offset = (i*l.groups + j)*k*n;
//                 // int col_offset = j*(l.size*l.size*(l.c/l.groups)) * l.out_w*l.out_h;
//                 im2col_gpu_offset(im.mem, im_offset, l.c/l.groups, l.h, l.w, 
//                           l.size, l.stride, l.pad, 
//                           b.mem,col_offset);
//             }
//             }
//             // 步骤2: 执行融合groups的GEMM
//             xgemm_gpu(0,0,m,n,k,1,
//                     a, 0, k, // weights [n, k] (k = groups*(c/groups)*size²)
//                     (l.size == 1)?net.input_gpu:net.workspace_gpu,  i* l.groups * k * n, n, // im2col [k, out_h*out_w*groups]
//                     1,
//                     c, i * n * m* l.groups, n,l.groups);
        
//     }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }


//******************优化删除group版本**********************
//******************优化删除group版本**********************
//******************优化删除group版本**********************
// forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//     if(l.binary){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c*l.size*l.size, l.binary_weights_gpu.mem); // 移除了 groups 除法
//         swap_binary_gpu(&l);
//     }

//     if(l.xnor){
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c*l.size*l.size, l.binary_weights_gpu.mem); // 移除了 groups 除法
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }

//     int i;
//     int m = l.n; // groups=1，直接等于 l.n
//     int k = l.size*l.size*l.c; // groups=1，直接等于 l.c*size²
//     int n = l.out_w*l.out_h;
//     for(i = 0; i < l.batch; ++i){
//         cl_mem_ext a = l.weights_gpu; // j=0，权重偏移为0
//         cl_mem_ext b = net.workspace_gpu;
//         cl_mem_ext c = l.output_gpu;
//         cl_mem_ext im = net.input_gpu;

//         if (l.size == 1){
//             b = im;
//             // 输入偏移为 i*l.c*l.h*l.w，输出偏移为 i*n*m
//             xgemm_gpu(0,0,m,n,k,1,a,0,k,b,i*l.c*l.h*l.w,n,1,c,i*n*m,n,l.batch);

//             //删除batch循环
//             // xgemm_gpu(0,0,m,n*l.batch,k,1,a,0,k,b,0,n,1,c,0,n*m,l.batch);
//         } else {
//             // 输入偏移为 i*l.c*l.h*l.w
//             // im2col_gpu_offset(im.mem, i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, b.mem,i*k*n);  设置了offsetB会出错，所以使用原始im2col_gpu
//             im2col_gpu(im.mem, i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, b.mem);

//             //删除batch循环
//             //im2col_gpu(im.mem, 0, l.c, l.h, l.w, l.size, l.stride, l.pad, b.mem);



//             // 直接使用 workspace 的起始地址，输出偏移为 i*n*m
//             xgemm_gpu(0,0,m,n,k,1,a,0,k,b,0,n,1,c,i*n*m,n,l.batch);

//             //删除batch循环
//             // xgemm_gpu(0,0,m,n*l.batch,k,1,a,0,k,b,0,n,1,c,0,n*m,l.batch);

//             //fx_mec cl版本
//             //xim2col_mec_gpu(im,i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
//             //xgemm_mec_gpu(0,0,a,b,c,l.h,l.size, l.c,l.stride,m,1,l.pad);
            
//             //fx_im2col cl版本
//             // xim2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//             // xgemm_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
//         }
//     }
//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
//     if(l.binary || l.xnor) swap_binary_gpu(&l);
// }

// void forward_convolutional_layer_gpu(convolutional_layer l, network net)
// {
//     fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1);

//     if (l.binary) {
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//     }

//     if (l.xnor) {
//         binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
//         swap_binary_gpu(&l);
//         binarize_gpu(net.input_gpu.mem, l.c * l.h * l.w * l.batch, l.binary_input_gpu.mem);
//         net.input_gpu = l.binary_input_gpu;
//     }
//     int i;
//     int m = l.n / l.groups;  // Number of output channels per group
//     int k = l.size * l.size * l.c / l.groups;  // Input channels per group
//     int n = l.out_w * l.out_h;  // Output height * width

//     // Process each group
    
//     for (int j = 0; j < l.groups; ++j) {
//         for(i = 0; i < l.batch; ++i){
//         cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
//         cl_mem_ext b = net.workspace_gpu;
//         cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
//         cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
//         if (l.size == 1){
//                 b = im;
//             } else {
//                 // im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
//                 im2col_gpu_offset(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem,(i*l.groups + j)*k*n);
//             }

//         }

        
// \        gemm_offset_gpu_strided_batched(0, 0, m, n, k,
//                                 1,
//                                 l.weights_gpu, j * l.nweights / l.groups, k, 0,
//                                 (l.size == 1)?net.input_gpu:net.workspace_gpu, j * k * n, n, l.groups * k * n,
//                                 1,
//                                 l.output_gpu, j * (m * n), n, (m * n * l.groups),
//                                 l.batch);
//     }

//     if (l.batch_normalize) {
//         forward_batchnorm_layer_gpu(l, net);
//     } else {
//         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w * l.out_h);
//     }

//     activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);

//     if (l.binary || l.xnor) {
//         swap_binary_gpu(&l);
//     }
// }

//******************fx promax batch版**********************
//******************fx promax batch版**********************
//******************fx promax batch版**********************
void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1);
    if (l.binary)
    {
        binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
        swap_binary_gpu(&l);
    }

    if (l.xnor)
    {
        binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu.mem);
        swap_binary_gpu(&l);
        binarize_gpu(net.input_gpu.mem, l.c * l.h * l.w * l.batch, l.binary_input_gpu.mem);
        net.input_gpu = l.binary_input_gpu;
    }

    int i, j;
    int m = l.n / l.groups;
    int k = l.size * l.size * l.c / l.groups;
    int n = l.out_w * l.out_h;
    // for(i = 0; i < l.batch; ++i){
    //     for(j = 0; j < l.groups; ++j){
    //         cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
    //         cl_mem_ext b = net.workspace_gpu;
    //         cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
    //         cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

    //         if (l.size == 1){
    //             b = im;
    //             gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n);
    //             //  xgemm_mec_gpu(0,0,a,b,c,l.h,l.size, l.c/l.groups,l.stride,m,1,l.pad);
    //         } else {
    //             im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
    //             gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
    //         }
    //     }
    // }

    cl_mem_ext a = l.weights_gpu;
    cl_mem_ext b = net.workspace_gpu;  
    cl_mem_ext c = l.output_gpu;
    cl_mem_ext im = net.input_gpu;
    xim2col_mec_gpu(im, 0, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, l.batch * l.groups, b);
    xgemm_mec_gpu(0, 0, a, b, c, l.h, l.size, l.c / l.groups, l.stride, m, l.batch, l.groups, l.pad);

    // printf(">>>fangxin>>> batch: %d, groups:%d\n",l.batch, l.groups);

    if (l.batch_normalize)
    {
        forward_batchnorm_layer_gpu(l, net);
    }
    else
    {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w * l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
    // if(l.dot > 0) dot_error_gpu(l);
    if (l.binary || l.xnor)
        swap_binary_gpu(&l);
}

void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    dim2 dimN;
    dimN = opencl_gridsize((const int) n);

    opencl_kernel(opencl_smooth_kernel[opencl_device_id_t], dimN, 16, &l.output_gpu.mem, sizeof(cl_mem), &n, sizeof(cl_int), &l.w, sizeof(cl_int), &l.h, sizeof(cl_int), &l.c, sizeof(cl_int), &size, sizeof(cl_int), &rate, sizeof(cl_float), &l.delta_gpu.mem, sizeof(cl_mem));
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }

    //constrain_gpu(l.outputs*l.batch, 1, net.delta_gpu, 1);

    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    cl_mem_ext original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;

    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            cl_mem_ext a = l.delta_gpu; // + (i*l.groups + j)*m*k;
            cl_mem_ext b = net.workspace_gpu;
            cl_mem_ext c = l.weight_updates_gpu; // + j*l.nweights/l.groups;

            cl_mem_ext im  = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            cl_mem_ext imd = net.delta_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
            gemm_offset_gpu(0,1,m,n,k,1,a,(i*l.groups + j)*m*k,k,b,0,k,1,c,j*l.nweights/l.groups,n);

            if (net.delta_gpu.ptr) {
                if (l.binary || l.xnor) swap_binary_gpu(&l);
                a = l.weights_gpu; // + j*l.nweights/l.groups;
                b = l.delta_gpu; // + (i*l.groups + j)*m*k;
                c = net.workspace_gpu;

                if (l.size == 1) {
                    c = imd;
                    gemm_offset_gpu(1,0,n,k,m,1,a,j*l.nweights/l.groups,n,b,(i*l.groups + j)*m*k,k,0,c,(i*l.groups + j)*l.c/l.groups*l.h*l.w,k);
                }
                else {
                    gemm_offset_gpu(1,0,n,k,m,1,a,j*l.nweights/l.groups,n,b,(i*l.groups + j)*m*k,k,0,c,0,k);
                }

                if (l.size != 1) {
                    col2im_gpu(net.workspace_gpu.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd.mem);
                }
                if(l.binary || l.xnor) {
                    swap_binary_gpu(&l);
                }
            }
            if(l.xnor) gradient_array_offset_gpu(original_input, i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu);
        }
    }
}

void pull_convolutional_layer(layer l)
{
    opencl_pull_array(l.weights_gpu, l.weights, l.nweights);
    opencl_pull_array(l.biases_gpu, l.biases, l.n);
    opencl_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    opencl_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_pull_array(l.scales_gpu, l.scales, l.n);
        opencl_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    opencl_push_array(l.weights_gpu, l.weights, l.nweights);
    opencl_push_array(l.biases_gpu, l.biases, l.n);
    opencl_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    opencl_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_push_array(l.scales_gpu, l.scales, l.n);
        opencl_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void adam_update_gpu(cl_mem_ext w, cl_mem_ext d, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate/batch, eps, t);
    fill_gpu(n, 0, d, 1);
}

void update_convolutional_layer_gpu(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu.ptr){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu.ptr){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

#endif // GPU