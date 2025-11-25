#ifdef GPU
// void xim2col_mec_gpu(cl_mem_ext im, int offset,
//                      int channels, int height, int width,
//                      int ksize, int stride, int pad, cl_mem_ext data_col);
void xim2col_mec_gpu(cl_mem_ext im, int offset,
                    int channels, int height, int width,
                    int ksize, int stride, int pad, int batch,cl_mem_ext data_col);
// void xgemm_mec_gpu(int TA, int TB,
//                    cl_mem_ext kerne_mec, cl_mem_ext img2col, cl_mem_ext output_mec,
//                    int img_size, int kernel_size, int channels, int stride,
//                    int out_channels, int batch, int pad);
void xgemm_mec_gpu(int TA, int TB,
    cl_mem_ext kerne_mec, cl_mem_ext img2col, cl_mem_ext output_mec,
    int img_size, int kernel_size, int channels, int stride,
    int out_channels, int batch, int groups, int pad);

void xim2col_gpu(cl_mem im, int offset,
                 int channels, int height, int width,
                 int ksize, int stride, int pad, cl_mem data_col);

// void xgemm_gpu(int TA, int TB,
//                cl_mem_ext img2col, cl_mem_ext kernel, cl_mem_ext output,
//                int img_size, int kernel_size, int channels, int stride,
//                int out_channels, int batch, int pad);

void xgemm_gpu(int TA, int TB, int M, int N, int K,
               float ALPHA,
               cl_mem_ext kernel, int a_offset, int a_ld,
               cl_mem_ext img2col, int b_offset, int b_ld,
               float BETA,
               cl_mem_ext output, int c_offset, int c_ld,int groups);

#endif