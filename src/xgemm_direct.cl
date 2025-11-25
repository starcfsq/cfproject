
static const char *const xgemm_direct_kernel_source = CONVERT_KERNEL_TO_STRING(
    __kernel void GlobalToLocalDirectA(const __global float *restrict agm, __local float *alm,
                                       const int a_ld, const int a_offset, const int kwg,
                                       const int a_transpose, const int a_conjugate,
                                       const int WGD, const int MDIMCD, const int NDIMCD, const int MDIMAD, const int PADA,
                                       const int KDIMAD, const int MWAD, const int KWAD) {
      int la0 = 0;
      int la1 = 0;
      if (MDIMCD == MDIMAD)
      {
        la0 = get_local_id(0);
        la1 = get_local_id(1);
      }
      else
      {
        const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
        la0 = tid % MDIMAD;
        la1 = tid / MDIMAD;
      }

      for (int _mia = 0; _mia < MWAD; _mia += 1)
      {

        for (int _kia = 0; _kia < KWAD; _kia += 1)
        {

          // Computes the indices for the global memory
          int mg = _mia + la0 * MWAD;
          int kg = _kia + la1 * KWAD;
          int idm = (a_transpose) ? mg + kwg : mg + get_group_id(0) * WGD;
          int idk = (a_transpose) ? kg + get_group_id(0) * WGD : kg + kwg;

          // Loads the data from global memory into the local memory
          alm[kg * (WGD + PADA) + mg] = agm[idk * a_ld + idm + a_offset];
          // printf("idm:%d, idk:%d,a_offset:%d, index:%d, Aresult:%0.0f \n", idm, idk, a_offset, idk * (a_ld / VWMD) + idm + (a_offset / VWMD), avec);
        }
      }
    }

    __kernel void GlobalToLocalDirectB(const __global float *restrict bgm, __local float *blm,
                                       const int b_ld, const int b_offset, const int kwg, const int kSizeN,
                                       const int b_transpose, const int b_conjugate,
                                       const int WGD, const int MDIMCD, const int NDIMCD, const int MDIMAD, const int NDIMBD, const int PADB,
                                       const int KDIMBD, const int KWBD, const int NWBD, const int NWID) {
      int lb0 = 0;
      int lb1 = 0;
      if (MDIMCD == NDIMBD)
      {
        lb0 = get_local_id(0);
        lb1 = get_local_id(1);
      }
      else
      {
        const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
        lb0 = tid % NDIMBD;
        lb1 = tid / NDIMBD;
      }

      for (int _kib = 0; _kib < KWBD; _kib += 1)
      {

        for (int _nib = 0; _nib < NWBD; _nib += 1)
        {

          // Computes the indices for the global memory
          int ng = _nib + lb0 * (NWBD);
          int kg = _kib + lb1 * KWBD;
          int idn = (b_transpose) ? ng + kwg : ng + get_group_id(1) * WGD;
          int idk = (b_transpose) ? kg + get_group_id(1) * WGD : kg + kwg;

          // Loads the data from global memory into the local memory
          blm[kg * (WGD + PADB) + ng] = bgm[idk * b_ld + idn + b_offset];
        }
      }
    }

    void GlobalToLocalCheckedA(const __global float *restrict agms, __local float *alm,
                               const int a_ld, const int a_offset, const int kwg,
                               const int a_transpose, const int a_conjugate,
                               const int kSizeM, const int kSizeK,
                               const int WGD, const int MDIMCD, const int NDIMCD, const int MDIMAD, const int PADA,
                               const int KDIMAD, const int MWAD, const int KWAD) {
      int la0 = 0;
      int la1 = 0;
      if (MDIMCD == MDIMAD)
      {
        la0 = get_local_id(0);
        la1 = get_local_id(1);
      }
      else
      {
        const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
        la0 = tid % MDIMAD;
        la1 = tid / MDIMAD;
      }

      for (int _mia = 0; _mia < MWAD; _mia += 1)
      {

        for (int _kia = 0; _kia < KWAD; _kia += 1)
        {

          // Computes the indices for the global memory
          int mg = _mia + la0 * MWAD;
          int kg = _kia + la1 * KWAD;
          int idm = (a_transpose) ? mg + kwg : mg + get_group_id(0) * WGD;
          int idk = (a_transpose) ? kg + get_group_id(0) * WGD : kg + kwg;

          // Loads the data from global memory into the local memory
          int condition = (a_transpose) ? (idm < kSizeK) && (idk < kSizeM) : (idm < kSizeM) && (idk < kSizeK);
          if (condition)
          {
            alm[kg * (WGD + PADA) + mg] = agms[idk * a_ld + idm + a_offset];
          }
          else
          {
            alm[kg * (WGD + PADA) + mg] = 0;
          }
        }
      }
    }

    // Same as above, but now for the B input matrix
    void GlobalToLocalCheckedB(const __global float *restrict bgms, __local float *blm,
                               const int b_ld, const int b_offset, const int kwg,
                               const int b_transpose, const int b_conjugate,
                               const int kSizeN, const int kSizeK,
                               const int WGD, const int MDIMCD, const int NDIMCD, const int NDIMBD, const int PADB,
                               const int KDIMBD, const int KWBD, const int NWBD, const int NWID) {
      int lb0 = 0;
      int lb1 = 0;
      if (MDIMCD == NDIMBD)
      {
        lb0 = get_local_id(0);
        lb1 = get_local_id(1);
      }
      else
      {
        const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
        lb0 = tid % NDIMBD;
        lb1 = tid / NDIMBD;
      }

      for (int _kib = 0; _kib < KWBD; _kib += 1)
      {

        for (int _nib = 0; _nib < NWBD; _nib += 1)
        {

          // Computes the indices for the global memory
          int ng = _nib + lb0 * NWBD;
          int kg = _kib + lb1 * KWBD;

          int idn = (b_transpose) ? ng + kwg : ng + get_group_id(1) * WGD;
          int idk = (b_transpose) ? kg + get_group_id(1) * WGD : kg + kwg;

          // Loads the data from global memory into the local memory
          int condition = (b_transpose) ? (idn < kSizeK) && (idk < kSizeN) : (idn < kSizeN) && (idk < kSizeK);
          if (condition)
          {
            blm[kg * (WGD + PADB) + ng] = bgms[idk * b_ld + idn + b_offset];
            // printf("KWBD:%d,kwg:%d,idn:%d, idk:%d, kg:%d, ng:%d, blmIndex:%d, result:%0.0f,k:%d\n", KWBD, kwg, idn, idk, kg, ng, kg * (WGD + PADB) + ng, result, k);
          }
          else
          {
            blm[kg * (WGD + PADB) + ng] = 0;
          }
        }
        // barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    float GlobalToPrivateDirectA(const __global float *restrict agms, const int _mi,
                                 const int a_ld, const int a_offset, const int idm, const int idk,
                                 const int a_transpose, const int a_conjugate) {
      const int a_index = (a_transpose) ? (idm + _mi) * a_ld + idk : idk * a_ld + (idm + _mi);
      return agms[a_index + a_offset];
    }

    // Same as above, but now for the B input matrix
    float GlobalToPrivateDirectB(const __global float *restrict bgms, const int _ni,
                                 const int b_ld, const int b_offset, const int idn, const int idk,
                                 const int b_transpose, const int b_conjugate) {
      const int b_index = (b_transpose) ? (idn + _ni) * b_ld + idk : idk * b_ld + (idn + _ni);
      return bgms[b_index + b_offset];
    }

    float GlobalToPrivateCheckedA(const __global float *restrict agms, const int _mi,
                                  const int a_ld, const int a_offset, const int idm, const int idk,
                                  const int a_transpose, const int a_conjugate,
                                  const int kSizeM) {
      float result;
      int index = idm + _mi;
      if (index < kSizeM)
      {
        const int a_index = (a_transpose) ? (idm + _mi) * a_ld + idk : idk * a_ld + (idm + _mi);
        result = agms[a_index + a_offset];
      }
      else
      {
        result = 0;
      }
      return result;
    }

    // Same as above, but now for the B input matrix
    float GlobalToPrivateCheckedB(const __global float *restrict bgms, const int _ni,
                                  const int b_ld, const int b_offset, const int idn, const int idk,
                                  const int b_transpose, const int b_conjugate,
                                  const int kSizeN) {
      float result;
      if (idn + _ni < kSizeN)
      {
        const int b_index = (b_transpose) ? (idn + _ni) * b_ld + idk : idk * b_ld + (idn + _ni);
        result = bgms[b_index + b_offset];
      }
      else
      {
        result = 0;
      }
      return result;
    }

    float LocalToPrivateDirectA(__local float *alm, const int _mi, const int kg,
                                const int a_transpose, const int WGD, const int PADA, const int MWID) {
      const int mg = _mi + get_local_id(0) * MWID;
      const int index = (a_transpose) ? mg * (WGD + PADA) + kg : kg * (WGD + PADA) + mg;
      return alm[index];
    }

    // Same as above, but now for the B input matrix
    float LocalToPrivateDirectB(__local float *blm, const int _ni, const int kg,
                                const int b_transpose, const int WGD, const int PADB, const int NWID) {
      const int ng = _ni + get_local_id(1) * NWID;
      const int index = kg * (WGD + PADB) + ng;
      return blm[index];
    }

    __kernel void StoreResultsDirect(__global float *cgm, const float c_value,
                                     const int _mi, const int _ni, const int idm, const int idn,
                                     const float alpha, const float beta,
                                     const int c_ld, const int c_offset, const int c_transpose) {
      // Determines the destination index
      int c_index = (c_transpose) ? (idm + _mi) * c_ld + (idn + _ni) : (idn + _ni) * c_ld + (idm + _mi);

      // The final multiplication with alpha (in case beta == 0)
      float result;
      if (beta == 0)
      {
        result = alpha * c_value;
      }
      // The final multiplication with alpha and the addition with beta*C
      else
      {
        result = alpha * c_value + beta * cgm[c_index + c_offset];
      }
      cgm[c_index + c_offset] = result;
    }

    __kernel void StoreResultsChecked(__global float *cgm, const float c_value,
                                      const int _mi, const int _ni, const int idm, const int idn,
                                      const int kSizeM, const int kSizeN,
                                      const float alpha, const float beta,
                                      const int c_ld, const int c_offset, const int c_transpose) {
      if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN)
      {

        // Deter_mines the destination index
        int c_index = (c_transpose) ? (idm + _mi) * c_ld + (idn + _ni) : (idn + _ni) * c_ld + (idm + _mi);

        // The final multiplication with alpha (in case beta == 0)
        float result;
        if (beta == 0)
        {
          result = alpha * c_value;
        }
        // The final multiplication with alpha and the addition with beta*C
        else
        {
          result = alpha * c_value + beta * cgm[c_index + c_offset];
        }
        cgm[c_index + c_offset] = result;
      }
    }

    __kernel void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                              const float alpha,
                              const float beta,
                              const __global float *restrict agm, int a_offset, const int a_ld,
                              const __global float *restrict bgm, int b_offset, const int b_ld,
                              __global float *cgm, const int c_offset, const int c_ld,
                              __local float *alm, __local float *blm,
                              const int a_transpose, const int b_transpose, const int c_transpose,
                              const int a_conjugate, const int b_conjugate,
                              const int WGD, const int MDIMCD, const int NDIMCD, const int MDIMAD, const int NDIMBD, const int KWID,
                              const int PADA, const int PADB, const int test) {
      // Helper parameters based on the above tuning parameters
      const int MWID = (WGD / MDIMCD);                   // Work per work-item (M-dimension)
      const int NWID = (WGD / NDIMCD);                   // Work per work-item (N-dimension)
      const int KDIMAD = ((MDIMCD * NDIMCD) / (MDIMAD)); // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
      const int KDIMBD = ((MDIMCD * NDIMCD) / (NDIMBD)); // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
      const int MWAD = (WGD / MDIMAD);                   // Amount of loads-per-thread for matrix A (M-dimension)
      const int KWAD = (WGD / KDIMAD);                   // Amount of loads-per-thread for matrix A (K-dimension)
      const int KWBD = (WGD / KDIMBD);                   // Amount of loads-per-thread for matrix B (K-dimension)
      const int NWBD = (WGD / NDIMBD);                   // Amount of loads-per-thread for matrix B (N-dimension)
      // Extra pointers to scalar versions of global memory
      const __global float *restrict agms = (const __global float *restrict)agm;
      const __global float *restrict bgms = (const __global float *restrict)bgm;

// Allocates workitem-private memory (registers)
#pragma promote_to_registers
      float apd[4]; // MWID
#pragma promote_to_registers
      float bpd[4]; // NWID
#pragma promote_to_registers
      float cpd[4 * 4]; // NWID * MWID

      // Initializes the accumulation registers

      for (int _mi = 0; _mi < MWID; _mi += 1)
      {

        for (int _ni = 0; _ni < NWID; _ni += 1)
        {
          cpd[_ni * MWID + _mi] = 0;
        }
      }

      // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
      // processes only the main parts: output blocks of WGD by WGD.
      const int idm = get_local_id(0) * MWID + get_group_id(0) * WGD;
      const int idn = get_local_id(1) * NWID + get_group_id(1) * WGD;

      // printf("idm:%d, idn:%d\n", idm, idn);

      if ((idm < (kSizeM / WGD) * WGD) && (idn < (kSizeN / WGD) * WGD))
      {
        int kwg = 0;

        // Loops over all complete workgroup tiles (K-dimension)

        for (; kwg < (kSizeK / WGD) * WGD; kwg += WGD)
        {

          // printf(">>>kwg1 : %d\n", kwg);
          // Loads data: off-chip --> local (matrix A and B)

          GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate,
                               WGD, MDIMCD, NDIMCD, MDIMAD, PADA, KDIMAD, MWAD, KWAD);
          GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, kSizeN, b_transpose, b_conjugate,
                               WGD, MDIMCD, NDIMCD, MDIMAD, NDIMBD, PADB, KDIMBD, KWBD, NWBD, NWID);

          barrier(CLK_LOCAL_MEM_FENCE);

          // Loops over all workitem tiles, unrolled by a factor KWID
          for (int pwi = 0; pwi < WGD; pwi += KWID)
          {

            for (int _pit = 0; _pit < KWID; _pit += 1)
            {
              int kg = pwi + _pit;

              // Loads data: local --> private (matrix A and B)

              for (int _mi = 0; _mi < MWID; _mi += 1)
              {
                apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose, WGD, PADA, MWID);
              }

              for (int _ni = 0; _ni < NWID; _ni += 1)
              {
                bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose, WGD, PADB, NWID);
              }

              // Performs the accumulation (Cpmd += Apmd * Bpmd)

              for (int _ni = 0; _ni < NWID; _ni += 1)
              {

                for (int _mi = 0; _mi < MWID; _mi += 1)
                {
                  cpd[_ni * MWID + _mi] += (apd[_mi] * bpd[_ni]);
                  if (test && apd[_mi] != 0 && bpd[_ni] != 0)
                  {
                    printf("kwg[1]: kwg:%d, idm:%d, idn:%d, _mi:%d, _ni:%d, a[%0.0f] * bpd[%0.0f] = c[%d]\n", kwg, idm, idn, _mi, _ni, apd[_mi], bpd[_ni], _ni * MWID + _mi);
                  }
                }
              }
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Loop over the remaining part (incomplete tile in K-dimension)
        for (; kwg < kSizeK; ++kwg)
        {

          // Loads data: off-chip --> private (matrix A and B)

          for (int _mi = 0; _mi < MWID; _mi += 1)
          {
            apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
          }

          for (int _ni = 0; _ni < NWID; _ni += 1)
          {
            bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)

          for (int _ni = 0; _ni < NWID; _ni += 1)
          {

            for (int _mi = 0; _mi < MWID; _mi += 1)
            {
              cpd[_ni * MWID + _mi] += (apd[_mi] * bpd[_ni]);
              if (test && apd[_mi] != 0 && bpd[_ni] != 0)
              {
                printf("kwg[2]: kwg:%d, idm:%d, idn:%d, _mi:%d, _ni:%d, a[%0.0f] * bpd[%0.0f] = c[%d]\n", kwg, idm, idn, _mi, _ni, apd[_mi], bpd[_ni], _ni * MWID + _mi);
              }
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
    // Stores a tile of results and performs the multiplication with alpha and beta
#pragma unrol
        for (int _ni = 0; _ni < NWID; _ni += 1)
        {
#pragma unrol
          for (int _mi = 0; _mi < MWID; _mi += 1)
          {
            StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                               alpha, beta, c_ld, c_offset, c_transpose);
          }
        }
      }

      // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
      else
      {
        int kwg = 0;

        // Loops over all complete workgroup tiles (K-dimension)
        for (; kwg < (kSizeK / WGD) * WGD; kwg += WGD)
        {

          // printf(">>>kwg3 : %d \n", kwg);
          // Loads data: off-chip --> local (matrix A and B)
          GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK,
                                WGD, MDIMCD, NDIMCD, MDIMAD, PADA, KDIMAD, MWAD, KWAD);

          GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK,
                                WGD, MDIMCD, NDIMCD, NDIMBD, PADB, KDIMBD, KWBD, NWBD, NWID);
          barrier(CLK_LOCAL_MEM_FENCE);

          // Loops over all workitem tiles, unrolled by a factor KWID
          for (int pwi = 0; pwi < WGD; pwi += KWID)
          {

            for (int _pit = 0; _pit < KWID; _pit += 1)
            {
              int kg = pwi + _pit;

              // Loads data: local --> private (matrix A and B)

              for (int _mi = 0; _mi < MWID; _mi += 1)
              {
                apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose, WGD, PADA, MWID);
              }

              for (int _ni = 0; _ni < NWID; _ni += 1)
              {
                bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose, WGD, PADB, NWID);
              }

              // Performs the accumulation (C += A * B)

              for (int _ni = 0; _ni < NWID; _ni += 1)
              {

                for (int _mi = 0; _mi < MWID; _mi += 1)
                {
                  cpd[_ni * MWID + _mi] += (apd[_mi] * bpd[_ni]);
                  if (test && apd[_mi] != 0 && bpd[_ni] != 0)
                  {
                    // printf("kwg[3]: kwg:%d, idm:%d, idn:%d, _mi:%d, _ni:%d, a[%0.0f] * bpd[%0.0f] = c[%d]\n", kwg, idm, idn, _mi, _ni, apd[_mi], bpd[_ni], _ni * MWID + _mi);
                  }
                }
              }
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Loop over the remaining part (incomplete tile in K-dimension)
        for (; kwg < kSizeK; ++kwg)
        {

          // Loads data: off-chip --> private (matrix A and B)

          // printf(">>>kwg4:%d\n", kwg);

          for (int _mi = 0; _mi < MWID; _mi += 1)
          {
            apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
            // printf("remain kwg:%d , apd:%0.0f \n", kwg, apd[_mi]);
          }

          for (int _ni = 0; _ni < NWID; _ni += 1)
          {
            bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
          }

          // Performs the accumulation (C += A * B)

          for (int _ni = 0; _ni < NWID; _ni += 1)
          {

            for (int _mi = 0; _mi < MWID; _mi += 1)
            {
              cpd[_ni * MWID + _mi] += (apd[_mi] * bpd[_ni]);
              if (test && apd[_mi] != 0 && bpd[_ni] != 0)
              {
                // printf("kwg[4]: kwg:%d,idm:%d, idn:%d, _mi:%d, _ni:%d, a[%0.0f] * bpd[%0.0f] = c[%d]\n", kwg, idm, idn, _mi, _ni, apd[_mi], bpd[_ni], _ni * MWID + _mi);
              }
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Stores a tile of results and performs the multiplication with alpha and beta

        for (int _ni = 0; _ni < NWID; _ni += 1)
        {

          for (int _mi = 0; _mi < MWID; _mi += 1)
          {
            StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
                                alpha, beta, c_ld, c_offset, c_transpose);
          }
        }
      }
    }

    __kernel void xgemm_kernel(const int kSizeM, const int kSizeN, const int kSizeK,
                               const float arg_alpha, const float arg_beta,
                               const __global float *restrict agm, const int a_offset, const int a_ld,
                               const __global float *restrict bgm, int b_offset, const int b_ld,
                               __global float *cgm, int c_offset, const int c_ld,
                               const int a_transpose, const int b_transpose,
                               const int c_transpose, const int a_conjugate, const int b_conjugate,
                              //  const int kernel_size, const int channels, const int output_size, const int out_channels, 
                               const int test) {
      const int WGD = 32;   // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
      const int MDIMCD = 8; // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
      const int NDIMCD = 8; // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
      const int MDIMAD = 8; // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
      const int NDIMBD = 8; // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
      const int KWID = 2;   // Unroll factor of the WGD loop (smaller or equal than WGD)
      const int VWMD = 1;   // Vector width of matrices A and C
      const int VWND = 1;   // Vector width of matrix B
      const int PADA = 1;   // Local memory padding for matrix A
      const int PADB = 1;   // Local memory padding for matrix B

      __local float alm[32 * (32 + 1)]; // WGD * (WGD + PADA)
      __local float blm[32 * (32 + 1)]; // WGD * (WGD + PADB)

      // const int b = get_global_id(2);

      // b_offset = b * kernel_size * kernel_size * output_size * output_size * channels;
      // c_offset = b * out_channels * output_size * output_size;
      XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
                  agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
                  alm, blm, a_transpose, b_transpose, c_transpose, a_conjugate, b_conjugate,
                  WGD, MDIMCD, NDIMCD, MDIMAD, NDIMBD, KWID, PADA, PADB, test);
    }

    // =================================================================================================
);
// End of the C++11 raw string literal

// =================================================================================================
static const char *const xim2col_kernel_source = CONVERT_KERNEL_TO_STRING(
    __kernel void xim2col_gpu_kernel(int n, __global float *data_im,
                                    int height, int width, int ksize,
                                    int pad,
                                    int stride,
                                    int height_col, int width_col,
                                    __global float *data_col) {
      // printf("\n...%d...\n", n);
      int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
      for (; index < n; index += get_global_size(1) * get_global_size(0))
      {
        // printf("\n--%d\n", index);
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        int data_col_offset = (channel_out * height_col + h_out) * width_col + w_out;
        int data_im_offset = (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; ++i)
        {
          for (int j = 0; j < ksize; ++j)
          {
            int h = h_in + i;
            int w = w_in + j;

            data_col[data_col_offset] = (h >= 0 && w >= 0 && h < height && w < width) ? data_im[data_im_offset + i * width + j] : 0;

            // data_col[data_col_offset] = data_im[data_im_offset + i * width + j];
            // printf("%d -[%0.0f] , ", data_col_offset, data_col[data_col_offset]);
            data_col_offset += height_col * width_col;
          }
        }
      }
    }

);