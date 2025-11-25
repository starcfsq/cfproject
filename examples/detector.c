#include <jiaoben.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "darknet.h"
#include "system.h"
#include "image.h"
#include <stdio.h>
#include "opencl.h"
#ifdef WIN32
#include "unistd\dirent.h"
#else
#include <dirent.h>
#endif

#ifdef WIN32
#include "unistd\unistd.h"
#else
#include <unistd.h>
#endif

#include <sys/stat.h>
#define class temp
// #ifdef WIN32
// __declspec(thread) int opencl_device_id_t;

// #else
// __thread int opencl_device_id_t;

// #endif



typedef struct
{
    network *net;
    char **paths;
    int start_idx;
    int num_images;
    int nthreads_images;
    float thresh;
    float nms;
    int classes;
    int coco;
    int imagenet;
    FILE *fp;
    FILE **fps;
    int *map;
    int device_id;
    double start_time;
    double end_time;
    image *val;
    image *val_resized;
    image *buf;
    image *buf_resized;
    pthread_t *thr;
    load_args args;
     // 为自适应新增：
    double avg_time_per_img;  // 单张图片的推理耗时（秒）
    int measured;             // 是否已测量过（例如达到一定批次数后记录）
    int batch_counter;        // 当前批次计数（用于采样）
} thread_args;

struct stat st;

static int coco_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = (network **)calloc(ngpus, sizeof(network *));

    srand(time(0));
    int seed = rand();
    int i;
    for (i = 0; i < ngpus; ++i)
    {
        srand(seed);
#ifdef GPU
        if (gpu_index >= 0)
        {
            opencl_set_device(i);
        }
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
#ifndef BENCHMARK
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
#endif
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    // int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    // args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
#ifdef LOSS_ONLY
    double time = what_time_is_it_now();
#else
    double time;
#endif
    int count = 0;

    if (count == 0)
    {
#ifdef GPU
        if (gpu_index >= 0)
        {
            if (ngpus != 1)
                sync_nets(nets, ngpus, 0);
        }
#endif
        char buff[256];
        sprintf(buff, "%s/%s.start.conv.weights", backup_directory, base);
        save_weights(net, buff);
    }

    int max_size = ((net->w + net->h) / 2);

    // while(i*imgs < N*120){
    while (get_current_batch(net) < net->max_batches)
    {
        if (l.random && count++ % 10 == 0)
        {
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("Resizing\n");
#endif
            int dim = max_size - ((rand() % 8) * 32);
#ifdef BENCHMARK
            dim = 608;
#endif
            if (get_current_batch(net) + 200 > net->max_batches)
                dim = max_size;
            if (net->w < dim || net->h < dim)
                dim = max_size;
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("%d\n", dim);
#endif
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

#pragma omp parallel for
            for (i = 0; i < ngpus; ++i)
            {
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
#ifndef LOSS_ONLY
        time = what_time_is_it_now();
#endif
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */
#ifndef LOSS_ONLY
        printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);
#endif
#ifndef LOSS_ONLY
        time = what_time_is_it_now();
#endif
        float loss = 0;
#ifdef GPU
        if (gpu_index >= 0)
        {
            if (ngpus == 1)
            {
                loss = train_network(net, train);
            }
            else
            {
                loss = train_networks(nets, ngpus, train, 4);
            }
        }
        else
        {
            loss = train_network(net, train);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0)
            avg_loss = loss;
        avg_loss = avg_loss * .9 + loss * .1;

        i = get_current_batch(net);
#ifdef LOSS_ONLY
        printf("%lf\t%f\n", what_time_is_it_now() - time, loss);
#else
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now() - time, i * imgs);
#endif
#ifdef GPU
        if (loss != loss && gpu_index >= 0)
        {
            opencl_deinit(gpusg, ngpusg);
        }
#endif
        if (loss != loss)
        {
            printf("NaN LOSS detected! No possible to continue!\n");
            exit(-7);
        }
        if (i % 100 == 0)
        {
#ifdef GPU
            if (gpu_index >= 0)
            {
                if (ngpus != 1)
                    sync_nets(nets, ngpus, 0);
            }
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if (i % 10000 == 0 || (i < 1000 && i % 100 == 0))
        {
#ifdef GPU
            if (gpu_index >= 0)
            {
                if (ngpus != 1)
                    sync_nets(nets, ngpus, 0);
            }
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
#ifdef GPU_STATS
        opencl_dump_mem_stat();
#endif
#ifdef BENCHMARK
        break;
#endif
    }
#ifdef GPU
    if (gpu_index >= 0)
    {
        if (ngpus != 1)
            sync_nets(nets, ngpus, 0);
    }
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    free(paths);
    free(plist);
    free(base);
    free(nets);
    free(options);
}

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c)
        p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for (i = 0; i < num_boxes; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0)
            xmin = 0;
        if (ymin < 0)
            ymin = 0;
        if (xmax > w)
            xmax = w;
        if (ymax > h)
            ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
                fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1)
            xmin = 1;
        if (ymin < 1)
            ymin = 1;
        if (xmax > w)
            xmax = w;
        if (ymax > h)
            ymax = h;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
                fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                        xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0)
            xmin = 0;
        if (ymin < 0)
            ymin = 0;
        if (xmax > w)
            xmax = w;
        if (ymax > h)
            ymax = h;

        for (j = 0; j < classes; ++j)
        {
            int class = j;
            if (dets[i].prob[class])
                fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[class],
                        xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf)
        map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n - 1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if (0 == strcmp(type, "coco"))
    {
        if (!outfile)
            outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    }
    else if (0 == strcmp(type, "imagenet"))
    {
        if (!outfile)
            outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    }
    else
    {
        if (!outfile)
            outfile = "comp4_det_test_";
        fps = (FILE **)calloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j)
        {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i = 0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = (image *)calloc(nthreads, sizeof(image));
    image *val_resized = (image *)calloc(nthreads, sizeof(image));
    image *buf = (image *)calloc(nthreads, sizeof(image));
    image *buf_resized = (image *)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t *)calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c * 2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    // args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t)
    {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for (i = nthreads; i < m + nthreads; i += nthreads)
    {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
        {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t)
        {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
        {
            char *path = paths[i + t - nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w * net->h * net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w * net->h * net->c, val_resized[t].data, 1, input.data + net->w * net->h * net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms)
            {
                if (l.nms_kind == DEFAULT_NMS)
                    do_nms_sort(dets, nboxes, l.classes, nms);
                else
                    diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            if (coco)
            {
                print_cocos(fp, path, dets, num, classes, w, h);
            }
            else if (imagenet)
            {
                print_imagenet_detections(fp, i + t - nthreads + 1, dets, num, classes, w, h);
            }
            else
            {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for (j = 0; j < classes; ++j)
    {
        if (fps)
            fclose(fps[j]);
    }
    if (coco)
    {
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


// 定义互斥锁和条件变量
pthread_mutex_t image_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t image_cond = PTHREAD_COND_INITIALIZER;

int global_image_index = 0;
int total_images = 60;
int buffer_size = 60; // 缓冲区大小
image *image_buffer;
image *image_resized_buffer;
int buffer_count = 0;
int finish =0;
int stop_cpu_inference = 0;
int count_gpu=0;

// 假设设备总数不变：
#define TOTAL_DEVICE 2
// 全局数组，利用互斥锁保护（或者在动态调度时已确保所有设备都测量到值）：
double device_speed[TOTAL_DEVICE] = {0};

// 假设动态调度的阈值：当剩余图片少于 TOTAL_IMAGES * 0.2 时进入动态调度阶段
#define DYNAMIC_THRESHOLD (total_images * 0.2)

// 假设每个设备允许的 batch_size 范围为 [1, 8]
#define MIN_BATCH_SIZE 1
#define MAX_BATCH_SIZE 4

void *run_thread_async1(void *args) {
    thread_args *arg = (thread_args *)args;
    arg->start_time = what_time_is_it_now();
    // 从网络最后一层获取nms类型等参数
    layer l = arg->net->layers[arg->net->n - 1];

    // 按设备初始设置批次大小（例如：CPU:2, GPU:4）
    int batch_size = (arg->device_id == 0) ? CPU_BATCH : GPU_BATCH;
    arg->measured = 0;
    arg->batch_counter = 0;

    while (1) {
        pthread_mutex_lock(&image_mutex);
        int remaining_images = total_images - (global_image_index - buffer_count);

        // 1.如果图片加载完并且缓冲区为空，则退出
        if (finish == 1 && buffer_count == 0) {
            pthread_mutex_unlock(&image_mutex);
            break;
        }

        // 动态调度阶段：当剩余图片较少时开始根据设备平均速度动态调整批次分配
        if (remaining_images <= DYNAMIC_THRESHOLD && arg->measured && device_speed[0] > 0 && device_speed[1] > 0) {
            // 假设 device_speed[0] 为 CPU 平均每张耗时, device_speed[1] 为 GPU 的
            double T_cpu = device_speed[0];
            double T_gpu = device_speed[1];

            int desired;
            if (arg->device_id == 0) {
                // 分配公式：N_cpu = round(remaining_images * (T_gpu)/(T_cpu + T_gpu)) clean
                desired = (int)round((remaining_images * T_gpu) / (T_cpu + T_gpu));
            } else {
                desired = remaining_images - (int)round((remaining_images * T_gpu) / (T_cpu + T_gpu));
            }
            // 保证批次大小在允许范围内
            if (desired < MIN_BATCH_SIZE)
                desired = MIN_BATCH_SIZE;
            if (desired > MAX_BATCH_SIZE)
                desired = MAX_BATCH_SIZE;
            if(desired>=batch_size)
                desired=batch_size;
          
            if (batch_size != desired) {
                batch_size = desired;
                set_batch_network(arg->net, batch_size);
                printf("Dynamic scheduling: device_id:%d, remaining:%d, new batch size:%d\n",
                       arg->device_id, remaining_images, batch_size);
            }
        }
        // 如果已完成加载但剩余图片数量不足于当前批次，则使用当前剩余的图片处理最后一批
            if (finish == 1 && buffer_count < batch_size) {
                batch_size = buffer_count;
                set_batch_network(arg->net, batch_size);
            }
            // 如果是CPU设备且剩余图片过少则提前退出，避免落后 GPU 太多
            // if (remaining_images <= 2 && arg->device_id == 0) {
            //     pthread_mutex_unlock(&image_mutex);
            //     printf("Stopping CPU inference as remaining images <= %d\n", 2);
            //     return 0;
            // }

        // 如果图片加载未完成且缓冲区图片数小于batch_size，则等待加载
        while (buffer_count < batch_size && finish == 0) {
            pthread_cond_wait(&image_cond, &image_mutex);
        }

        if (buffer_count >= batch_size) {
            int batch_start_idx = global_image_index - buffer_count;
            buffer_count -= batch_size;
            // 通知其他线程继续加载图片
            pthread_cond_signal(&image_cond);
            pthread_mutex_unlock(&image_mutex);

            // 分配内存加载当前批次图片及其缩放版本
            image *batch_images = (image *)calloc(batch_size, sizeof(image));
            image *batch_resized_images = (image *)calloc(batch_size, sizeof(image));
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch_start_idx + i;
                batch_images[i] = image_buffer[idx % buffer_size];
                batch_resized_images[i] = image_resized_buffer[idx % buffer_size];
            }
            
            // 提取输入数据准备批量推理
            float *X = (float *)calloc(batch_size * arg->net->h * arg->net->w * arg->net->c, sizeof(float));
            for (int i = 0; i < batch_size; ++i) {
                memcpy(X + i * arg->net->h * arg->net->w * arg->net->c,
                       batch_resized_images[i].data,
                       arg->net->h * arg->net->w * arg->net->c * sizeof(float));
            }

            // 执行推理
            opencl_device_id_t = arg->device_id;
            double batch_start_time = what_time_is_it_now();
            network_predict(arg->net, X);
            double batch_end_time = what_time_is_it_now();
            double batch_inference_time = batch_end_time - batch_start_time;
            printf("******Batch inference time: %f seconds, device_id: %d, batch_size: %d *****\n",
                   batch_inference_time, arg->device_id, batch_size);

            // 累计批次数，为采样计算平均单图耗时做准备（例如取第三个批次作为采样结果）
            arg->batch_counter++;
            if (!arg->measured && arg->batch_counter >= 2) {
                arg->avg_time_per_img = batch_inference_time / batch_size;
                arg->measured = 1;
                // 将该设备测得的平均速度写入全局数组
                pthread_mutex_lock(&image_mutex);
                device_speed[arg->device_id] = arg->avg_time_per_img;
                pthread_mutex_unlock(&image_mutex);
                printf("Measured device_id:%d avg time per image: %f seconds\n",
                       arg->device_id, arg->avg_time_per_img);
            }

            // 处理推理结果（这里与原来的处理流程保持一致）
            for (int i = 0; i < batch_size; ++i) {
                image im = batch_images[i];
                image resized_im = batch_resized_images[i];
                int w = im.w;
                int h = im.h;
                int nboxes = 0;
                detection *dets = get_network_boxes_batch(arg->net, w, h, arg->thresh, 0.5, arg->map, 0, &nboxes, i);

                if (arg->nms) {
                    if (l.nms_kind == DEFAULT_NMS)
                        do_nms_sort(dets, nboxes, arg->classes, arg->nms);
                    else
                        diounms_sort_y4(dets, nboxes, arg->classes, arg->nms, l.nms_kind, l.beta_nms);
                }

                if (arg->coco) {
                    print_cocos(arg->fp, arg->paths[batch_start_idx + i], dets, nboxes, arg->classes, w, h);
                } else if (arg->imagenet) {
                    print_imagenet_detections(arg->fp, 0, dets, nboxes, arg->classes, w, h);
                } else {
                    print_detector_detections(arg->fps, basecfg(arg->paths[batch_start_idx + i]),
                                              dets, nboxes, arg->classes, w, h);
                }
                free_detections(dets, nboxes);
                free_image(im);
                free_image(resized_im);
            }

            free(X);
            free(batch_images);
            free(batch_resized_images);
        }
        else {
            // 如果缓冲区图片不足但加载已完成，则处理最后剩余的小批次
            if (finish == 1 && buffer_count > 0) {
                int batch_start_idx = global_image_index - buffer_count;
                batch_size = buffer_count;
                set_batch_network(arg->net, batch_size);
                buffer_count = 0;
                pthread_cond_signal(&image_cond);
                pthread_mutex_unlock(&image_mutex);

                image *batch_images = (image *)calloc(batch_size, sizeof(image));
                image *batch_resized_images = (image *)calloc(batch_size, sizeof(image));
                for (int i = 0; i < batch_size; ++i) {
                    int idx = batch_start_idx + i;
                    batch_images[i] = image_buffer[idx % buffer_size];
                    batch_resized_images[i] = image_resized_buffer[idx % buffer_size];
                }

                float *X = (float *)calloc(batch_size * arg->net->h * arg->net->w * arg->net->c, sizeof(float));
                for (int i = 0; i < batch_size; ++i) {
                    memcpy(X + i * arg->net->h * arg->net->w * arg->net->c,
                           batch_resized_images[i].data,
                           arg->net->h * arg->net->w * arg->net->c * sizeof(float));
                }
                opencl_device_id_t = arg->device_id;
                double batch_start_time = what_time_is_it_now();
                network_predict(arg->net, X);
                double batch_end_time = what_time_is_it_now();
                double batch_inference_time = batch_end_time - batch_start_time;
                printf("******Batch inference time: %f seconds, device_id: %d *****\n",
                       batch_inference_time, arg->device_id);

                for (int i = 0; i < batch_size; ++i) {
                    image im = batch_images[i];
                    image resized_im = batch_resized_images[i];
                    int w = im.w;
                    int h = im.h;
                    int nboxes = 0;
                    detection *dets = get_network_boxes_batch(arg->net, w, h, arg->thresh, 0.5, arg->map, 0, &nboxes, i);

                    if (arg->nms) {
                        if (l.nms_kind == DEFAULT_NMS)
                            do_nms_sort(dets, nboxes, arg->classes, arg->nms);
                        else
                            diounms_sort_y4(dets, nboxes, arg->classes, arg->nms, l.nms_kind, l.beta_nms);
                    }

                    if (arg->coco) {
                        print_cocos(arg->fp, arg->paths[batch_start_idx + i], dets, nboxes, arg->classes, w, h);
                    } else if (arg->imagenet) {
                        print_imagenet_detections(arg->fp, 0, dets, nboxes, arg->classes, w, h);
                    } else {
                        print_detector_detections(arg->fps, basecfg(arg->paths[batch_start_idx + i]),
                                                  dets, nboxes, arg->classes, w, h);
                    }

                    free_detections(dets, nboxes);
                    free_image(im);
                    free_image(resized_im);
                }
                free(X);
                free(batch_images);
                free(batch_resized_images);
            }
        }
    }
    return 0;
}


void *load_images(void *args) {
    thread_args *arg = (thread_args *)args;

    while (1) {
        pthread_mutex_lock(&image_mutex);
        // 如果缓冲区满，等待消费者线程消费
        while (buffer_count == buffer_size && !finish) {
            pthread_cond_wait(&image_cond, &image_mutex);
        }

        if (finish && buffer_count == 0) {
            pthread_mutex_unlock(&image_mutex);
            break;
        }

        if (global_image_index >= total_images) {
            finish=1;
            pthread_cond_broadcast(&image_cond); // 通知所有线程退出
            pthread_mutex_unlock(&image_mutex);
            
            break;
        }

        int image_idx = global_image_index;
        // pthread_mutex_unlock(&image_mutex);
        // pthread_mutex_lock(&image_mutex);
        arg->args.path = arg->paths[image_idx];
        arg->args.im = &image_buffer[image_idx % buffer_size];
        arg->args.resized = &image_resized_buffer[image_idx % buffer_size];
        pthread_mutex_unlock(&image_mutex); // 解锁，允许其他线程访问

        arg->thr[0] = load_data_in_thread(arg->args);
        // load_data_single(arg->args);  // 改为同步加载数据
        pthread_join(arg->thr[0], 0);
        
        pthread_mutex_lock(&image_mutex);
        buffer_count++;
        global_image_index+=1;
        pthread_cond_signal(&image_cond);
        pthread_mutex_unlock(&image_mutex);
    }

    return 0;
}
void *run_thread_async(void *args) {
    thread_args *arg = (thread_args *)args;
    arg->start_time = what_time_is_it_now();
    layer l = arg->net->layers[arg->net->n - 1];

    // 设定每个设备的批次大小
    int batch_size = (arg->device_id == 0) ? CPU_BATCH : GPU_BATCH; // CPU 2张图片, GPU 4张图片

    while (1) {
        pthread_mutex_lock(&image_mutex);
        int remaining_images = total_images - (global_image_index - buffer_count);
    
        // if (remaining_images <=2  && arg->device_id == 0) {
        //     pthread_mutex_unlock(&image_mutex);
        //     printf("Stopping CPU inference as remaining images <= %d\n", 2);
        //     return;
        // }
        // if (remaining_images <= 13  && batch_size!=1) {
        //     batch_size=1;
        //     set_batch_network(arg->net, batch_size);
        // }

        // 1. 如果图片加载完并且 buffer_count == 0，则退出
        if (finish == 1 && buffer_count == 0) {
            pthread_mutex_unlock(&image_mutex);
            break;
        }

        // 2. 如果图片加载未完成且 buffer_count 小于批次大小，则继续等待加载
        while (buffer_count < batch_size && finish == 0) {
            pthread_cond_wait(&image_cond, &image_mutex);
        }

        // 3. 处理最后一批图片：如果所有图片加载完成但 buffer_count < batch_size
        if (finish == 1 && buffer_count < batch_size) {
            batch_size = buffer_count;  // 最后一批，处理剩余的图片
            set_batch_network(arg->net, batch_size);
        }

        // 4. 如果 buffer_count >= batch_size，开始处理当前批次的图片
        if (buffer_count >= batch_size) {
            int batch_start_idx = global_image_index - buffer_count;
            buffer_count -= batch_size;  // 更新缓冲区图片计数
            pthread_cond_signal(&image_cond);  // 唤醒其他线程继续加载图片
            pthread_mutex_unlock(&image_mutex);

            // 批量数据准备
            image *batch_images = (image *)calloc(batch_size, sizeof(image));
            image *batch_resized_images = (image *)calloc(batch_size, sizeof(image));
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch_start_idx + i;
                batch_images[i] = image_buffer[idx % buffer_size];
                batch_resized_images[i] = image_resized_buffer[idx % buffer_size];
            }

            // 提取输入数据并进行批量推理
            float *X = (float *)calloc(batch_size * arg->net->h * arg->net->w * arg->net->c, sizeof(float));
            for (int i = 0; i < batch_size; ++i) {
                memcpy(X + i * arg->net->h * arg->net->w * arg->net->c, batch_resized_images[i].data, arg->net->h * arg->net->w * arg->net->c * sizeof(float));
            }

            // 执行推理
            opencl_device_id_t = arg->device_id;
            double batch_start_time = what_time_is_it_now();
            network_predict(arg->net, X);
            double batch_end_time = what_time_is_it_now();
            double batch_inference_time = batch_end_time - batch_start_time;
            printf("******Batch inference time: %f seconds device_id: %d *****\n", batch_inference_time, arg->device_id);

            // 处理推理结果
            for (int i = 0; i < batch_size; ++i) {
                image im = batch_images[i];
                image resized_im = batch_resized_images[i];
                int w = im.w;
                int h = im.h;
                int nboxes = 0;
                // detection *dets = get_network_boxes(arg->net, w, h, arg->thresh, .5, arg->map, 0, &nboxes);
                detection *dets = get_network_boxes_batch(arg->net, w, h, arg->thresh, .5, arg->map, 0, &nboxes, i);

                if (arg->nms) {
                    if (l.nms_kind == DEFAULT_NMS)
                        do_nms_sort(dets, nboxes, arg->classes, arg->nms);
                    else
                        diounms_sort_y4(dets, nboxes, arg->classes, arg->nms, l.nms_kind, l.beta_nms);
                }

                // 输出检测结果
                if (arg->coco) {
                    print_cocos(arg->fp, arg->paths[batch_start_idx + i], dets, nboxes, arg->classes, w, h);
                } else if (arg->imagenet) {
                    print_imagenet_detections(arg->fp, 0, dets, nboxes, arg->classes, w, h);
                } else {
                    print_detector_detections(arg->fps, basecfg(arg->paths[batch_start_idx + i]), dets, nboxes, arg->classes, w, h);
                }

                free_detections(dets, nboxes);
                free_image(im);
                free_image(resized_im);
            }

            free(X);
            free(batch_images);
            free(batch_resized_images);
        } else {
            // 如果 buffer_count 不足批次大小，但已经加载完所有图片，则继续处理
            if (finish == 1 && buffer_count > 0) {
                // 处理剩余的小批次
                int batch_start_idx = global_image_index - buffer_count;
                batch_size = buffer_count;  // 设置批次大小为剩余图片的数量
                set_batch_network(arg->net, batch_size);
                buffer_count = 0;
                pthread_cond_signal(&image_cond);  // 唤醒其他线程继续加载图片
                pthread_mutex_unlock(&image_mutex);

                // 批量数据准备
                image *batch_images = (image *)calloc(batch_size, sizeof(image));
                image *batch_resized_images = (image *)calloc(batch_size, sizeof(image));
                for (int i = 0; i < batch_size; ++i) {
                    int idx = batch_start_idx + i;
                    batch_images[i] = image_buffer[idx % buffer_size];
                    batch_resized_images[i] = image_resized_buffer[idx % buffer_size];
                }

                // 提取输入数据并进行批量推理
                // float *X = (float *)calloc(batch_size * batch_images[0].h * batch_images[0].w * 3, sizeof(float));
                // for (int i = 0; i < batch_size; ++i) {
                //     memcpy(X + i * batch_images[0].h * batch_images[0].w * 3, batch_resized_images[i].data, batch_images[0].h * batch_images[0].w * 3 * sizeof(float));
                // }
                float *X = (float *)calloc(batch_size * arg->net->h * arg->net->w * arg->net->c, sizeof(float));
                for (int i = 0; i < batch_size; ++i) {
                memcpy(X + i * arg->net->h * arg->net->w * arg->net->c, batch_resized_images[i].data, arg->net->h * arg->net->w * arg->net->c * sizeof(float));
                }
                // 执行推理
                opencl_device_id_t = arg->device_id;
                double batch_start_time = what_time_is_it_now();
                network_predict(arg->net, X);
                double batch_end_time = what_time_is_it_now();
                double batch_inference_time = batch_end_time - batch_start_time;
                printf("******Batch inference time: %f seconds device_id: %d *****\n", batch_inference_time, arg->device_id);

                // 处理推理结果
                for (int i = 0; i < batch_size; ++i) {
                    image im = batch_images[i];
                    image resized_im = batch_resized_images[i];
                    int w = im.w;
                    int h = im.h;
                    int nboxes = 0;
                    // detection *dets = get_network_boxes(arg->net, w, h, arg->thresh, .5, arg->map, 0, &nboxes);
                    detection *dets = get_network_boxes_batch(arg->net, w, h, arg->thresh, .5, arg->map, 0, &nboxes, i);

                    if (arg->nms) {
                        if (l.nms_kind == DEFAULT_NMS)
                            do_nms_sort(dets, nboxes, arg->classes, arg->nms);
                        else
                            diounms_sort_y4(dets, nboxes, arg->classes, arg->nms, l.nms_kind, l.beta_nms);
                    }

                    // 输出检测结果
                    if (arg->coco) {
                        print_cocos(arg->fp, arg->paths[batch_start_idx + i], dets, nboxes, arg->classes, w, h);
                    } else if (arg->imagenet) {
                        print_imagenet_detections(arg->fp, 0, dets, nboxes, arg->classes, w, h);
                    } else {
                        print_detector_detections(arg->fps, basecfg(arg->paths[batch_start_idx + i]), dets, nboxes, arg->classes, w, h);
                    }

                    free_detections(dets, nboxes);
                    free_image(im);
                    free_image(resized_im);
                }

                free(X);
                free(batch_images);
                free(batch_resized_images);
            }
        }
    }

    return 0;
}



void *run_thread(void *args) {
    thread_args *arg = (thread_args *)args;
    arg->start_time = what_time_is_it_now();
    layer l = arg->net->layers[arg->net->n - 1];

    while (1) {
        pthread_mutex_lock(&image_mutex);
        while (buffer_count == 0) {
            // if(finish == 1 || (stop_cpu_inference && arg->device_id != 0)){
            if(finish == 1){
                pthread_mutex_unlock(&image_mutex);
                break;}
            pthread_cond_wait(&image_cond, &image_mutex);
        }

        int remaining_images = total_images - (global_image_index - buffer_count);
        // if (remaining_images <= 4 && (arg->device_id == 0 || arg->device_id == 1 ||arg->device_id == 2||arg->device_id ==3)) {
        // if (remaining_images <= 4 && (arg->device_id != 8 || arg->device_id != 9)) {
        //     stop_cpu_inference = 1;
        //     pthread_mutex_unlock(&image_mutex);
        //     printf("Stopping CPU inference as remaining images <= %d\n", 4);
        //     return 0;
        // }

        if (global_image_index >= total_images && buffer_count == 0) {
            pthread_mutex_unlock(&image_mutex);
            break;
        }

        int current_image_index = global_image_index - buffer_count;
        image im = image_buffer[current_image_index % buffer_size];
        image resized_im = image_resized_buffer[current_image_index % buffer_size];
        buffer_count--;
        pthread_cond_signal(&image_cond);
        pthread_mutex_unlock(&image_mutex);

        char *id = basecfg(arg->paths[current_image_index]);
        float *X = resized_im.data;
        opencl_device_id_t = arg->device_id;

        double image_start_time = what_time_is_it_now();
        network_predict(arg->net, X);
        double image_end_time = what_time_is_it_now();

        double image_inference_time = image_end_time - image_start_time;
        // count_gpu=count_gpu+(arg->device_id==1);
        printf("******Image %d inference time: %f seconds device_id: %d *****\n", current_image_index, image_inference_time, arg->device_id);

        int w = im.w;
        int h = im.h;
        int nboxes = 0;
        detection *dets = get_network_boxes(arg->net, w, h, arg->thresh, .5, arg->map, 0, &nboxes);

        if (arg->nms) {
            if (l.nms_kind == DEFAULT_NMS)
                do_nms_sort(dets, nboxes, arg->classes, arg->nms);
            else
                diounms_sort_y4(dets, nboxes, arg->classes, arg->nms, l.nms_kind, l.beta_nms);
        }

        if (arg->coco) {
            print_cocos(arg->fp, arg->paths[current_image_index], dets, nboxes, arg->classes, w, h);
        } else if (arg->imagenet) {
            print_imagenet_detections(arg->fp, 0, dets, nboxes, arg->classes, w, h);
        } else {
            print_detector_detections(arg->fps, id, dets, nboxes, arg->classes, w, h);
        }

        free_detections(dets, nboxes);
        free(id);
        free_image(im);
        free_image(resized_im);

       
    }

    return 0;
}

void validate_detector_async(char *datacfg, char *cfgfile, char *weightfile, char *outfile) {
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    // 设置设备总数（CPU 和 GPU）
    int total_device = 2;
    network *nets[total_device];

    // 为每个设备创建网络实例
    for (int d = 0; d < total_device; ++d) {
        opencl_device_id_t = d;
        nets[d] = load_network(cfgfile, weightfile, 0);

        // 根据设备性能调整 batch 大小
        if (d == 0) { 
            set_batch_network(nets[d], CPU_BATCH); // 假设 CPU 批大小为 2
        } else if (d == 1) { 
            set_batch_network(nets[d], GPU_BATCH); // 假设 GPU 批大小为 4
        }
    }

    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", nets[0]->learning_rate, nets[0]->momentum, nets[0]->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);
    layer l = nets[0]->layers[nets[0]->n - 1];
    int classes = l.classes;
    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        if (!fp) {
            perror("Error opening file");
            exit(1);
        }
        fprintf(fp, "[\n");
        coco = 1;
    } else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE **)calloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int nthreads = total_device; // 使用与设备数量相同的线程
    // int nthreads = 1; // 使用与设备数量相同的线程

    pthread_t load_thread;
    pthread_t threads[nthreads];
    thread_args load_thread_args;
    thread_args thread_args_array[nthreads];

    image_buffer = (image *)calloc(buffer_size, sizeof(image));
    image_resized_buffer = (image *)calloc(buffer_size, sizeof(image));
    double start = what_time_is_it_now();

    // 创建加载线程
    load_thread_args.paths = paths;
    load_thread_args.args = (load_args){0};
    load_thread_args.args.w = nets[0]->w;
    load_thread_args.args.h = nets[0]->h;
    load_thread_args.args.type = LETTERBOX_DATA;
    load_thread_args.thr = (pthread_t *)calloc(1, sizeof(pthread_t));
    if (pthread_create(&load_thread, 0, load_images, &load_thread_args)) {
        perror("Load thread creation failed");
        exit(1);
    }

    // 创建推理线程
    //i改成1是GPU
    for (int i = 0; i < nthreads; i++) {
        thread_args_array[i].net = nets[i];
        thread_args_array[i].paths = paths;
        thread_args_array[i].thresh = .005;
        thread_args_array[i].nms = .45;
        thread_args_array[i].classes = classes;
        thread_args_array[i].coco = coco;
        thread_args_array[i].imagenet = imagenet;
        thread_args_array[i].fp = fp;
        thread_args_array[i].fps = fps;
        thread_args_array[i].map = map;
        thread_args_array[i].device_id = i;

        // 为每个设备分配不同的批次缓冲区
        thread_args_array[i].val = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].val_resized = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].buf = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].buf_resized = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].thr = (pthread_t *)calloc(1, sizeof(pthread_t));

        if (pthread_create(&threads[i], 0, run_thread_async, &thread_args_array[i])) {
            perror("Inference thread creation failed");
            exit(1);
        }
    }

    // 等待加载线程完成
    pthread_join(load_thread, 0);

    // 等待推理线程完成 
    //i改成1是GPU
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[t], 0);
    }

    for (j = 0; j < classes; ++j) {
        if (fps) fclose(fps[j]);
    }

    if (coco) {
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }

    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}



void validate_detector_modify(char *datacfg, char *cfgfile, char *weightfile, char *outfile) {
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);
    //设置设备总数
    int total_device = 4;
    network *nets[total_device];
    for (int d = 0; d < total_device; ++d) {
        opencl_device_id_t = d;
        nets[d] = load_network(cfgfile, weightfile, 0);
        set_batch_network(nets[d], 1);
    }

    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", nets[0]->learning_rate, nets[0]->momentum, nets[0]->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);
    layer l = nets[0]->layers[nets[0]->n - 1];
    int classes = l.classes;
    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        if (!fp) {
            perror("Error opening file");
            exit(1);
        }
        fprintf(fp, "[\n");
        coco = 1;
    } else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE **)calloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int nthreads = total_device; // 使用2个线程

    pthread_t load_thread;
    pthread_t threads[nthreads];
    thread_args load_thread_args;
    thread_args thread_args_array[nthreads];

    image_buffer = (image *)calloc(buffer_size, sizeof(image));
    image_resized_buffer = (image *)calloc(buffer_size, sizeof(image));
    double start =what_time_is_it_now();
    // 创建加载线程
    load_thread_args.paths = paths;
    load_thread_args.args = (load_args){0};
    load_thread_args.args.w = nets[0]->w;
    load_thread_args.args.h = nets[0]->h;
    load_thread_args.args.type = LETTERBOX_DATA;
    load_thread_args.thr = (pthread_t *)calloc(1, sizeof(pthread_t));
    if (pthread_create(&load_thread, 0, load_images, &load_thread_args)) {
        perror("Load thread creation failed");
        exit(1);
    }

    // 创建推理线程
    for (int i = 0; i < nthreads; i++) {
        thread_args_array[i].net = nets[i];
        thread_args_array[i].paths = paths;
        thread_args_array[i].thresh = .005;
        thread_args_array[i].nms = .45;
        thread_args_array[i].classes = classes;
        thread_args_array[i].coco = coco;
        thread_args_array[i].imagenet = imagenet;
        thread_args_array[i].fp = fp;
        thread_args_array[i].fps = fps;
        thread_args_array[i].map = map;
        thread_args_array[i].device_id = i;
        thread_args_array[i].val = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].val_resized = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].buf = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].buf_resized = (image *)calloc(buffer_size, sizeof(image));
        thread_args_array[i].thr = (pthread_t *)calloc(1, sizeof(pthread_t));

        if (pthread_create(&threads[i], 0, run_thread, &thread_args_array[i])) {
            perror("Inference thread creation failed");
            exit(1);
        }
    }

    // 等待加载线程完成
    pthread_join(load_thread, 0);

    // 等待推理线程完成
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[t], 0);
    }

    for (j = 0; j < classes; ++j) {
        if (fps) fclose(fps[j]);
    }

    if (coco) {
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }

    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
    // printf("******GPU inference total images: %d  *****\n", count_gpu);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);
    //改为GPU
    opencl_device_id_t =1;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = (FILE**)calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    // int m = plist->size;
    // int m = total_images;
    int m = 60;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            double inference_start_time = what_time_is_it_now();
            network_predict(net, X);
            double inference_end_time = what_time_is_it_now();
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
            double inference_time = inference_end_time - inference_start_time;
            printf("***************************************Inference time for image %d: %f seconds\n",  i + t - nthreads, inference_time);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    // list *plist = get_paths("data/coco_val_5k.list");
    list *options = read_data_cfg(datacfg);
    char *test_images = option_find_str(options, "test", "data/test.list");
    list *plist = get_paths(test_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n - 1];

    int j, k;

    int m = plist->size;
    int i = 0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for (i = 0; i < m; ++i)
    {
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms)
        {
            if (l.nms_kind == DEFAULT_NMS)
                do_nms_sort(dets, nboxes, l.classes, nms);
            else
                diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (k = 0; k < nboxes; ++k)
        {
            if (dets[k].objectness > thresh)
            {
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j)
        {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for (k = 0; k < l.w * l.h * l.n; ++k)
            {
                float iou = box_iou(dets[k].bbox, t);
                if (dets[k].objectness > thresh && iou > best_iou)
                {
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if (best_iou > iou_thresh)
            {
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100. * correct / total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{   
    // opencl_device_id_t = 1;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms = .45;
    while (1)
    {
        if (filename)
        {
            strncpy(input, filename, 256);
        }
        else
        {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input)
                return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        int resize = im.w != net->w || im.h != net->h;
        image sized = resize ? letterbox_image(im, net->w, net->h) : im;
        // image sized = resize_image(im, net->w, net->h);
        // image sized2 = resize_max(im, net->w);
        // image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        // resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n - 1];

        float *X = sized.data;
        time = what_time_is_it_now();
        if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
        {
            network_predict(net, X);
        }
        if (l.type == YOLO4)
        {
            network_predict_y4(net, X);
        }
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now() - time);
        int nboxes = 0;
        detection *dets = 0;
        if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
        {
            dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        }
        if (l.type == YOLO4)
        {
            dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        }
        // printf("%d\n", nboxes);
        if (nms)
        {
            if (l.nms_kind == DEFAULT_NMS)
                do_nms_sort(dets, nboxes, l.classes, nms);
            else
                diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
        {
            draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes, 0);
        }
        if (l.type == YOLO4)
        {
            draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, 0);
        }
        free_detections(dets, nboxes);
        if (outfile)
        {
            save_image(im, outfile);
        }
        else
        {
            save_image(im, "predictions");
#ifdef OPENCV
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        if (resize)
            free_image(sized);
        if (filename)
            break;
    }
}
#ifndef __linux__
void test_ddetector(char *datacfg, char *cfgfile, char *weightfile, char *in_dir, float thresh, float hier_thresh, char *out_dir, int margin)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms = .45;
    char fname[1024];
    char ffname[1024];
    char ffoname[1024];

    struct dirent *de = NULL;

    while (1)
    {
        while (empty(in_dir))
        {
            usleep(100);
        }
        DIR *dr = opendir(in_dir);
        while ((de = readdir(dr)) != NULL)
        {
            // printf("%s\n", de->d_name);
            strcpy(fname, de->d_name);
            strcpy(ffname, in_dir);
            strcat(ffname, "/");
            strcat(ffname, fname);
            if (!exists(ffname, ".jpg"))
                continue;
            if (1)
            {
                strcpy(ffoname, out_dir);
                strcat(ffoname, "/");
                strcat(ffoname, fname);
                int len = strlen(ffoname) - 4;
                ffoname[len] = '\0';
                strncpy(input, ffname, 256);
            }
            else
            {
                printf("Enter Image Path: ");
                fflush(stdout);
                input = fgets(input, 256, stdin);
                if (!input)
                    continue;
                strtok(input, "\n");
            }
            off_t size = 0;
            off_t offs = 0;
            do
            {
                offs = size;
                stat(input, &st);
                size = st.st_size;
                if (offs != size)
                    usleep(10);
                else
                    break;
            } while (1);
            image im = load_image_color(input, 0, 0);
            if (im.w == 0 || im.h == 0)
            {
                remove(input);
                continue;
            }
            int resize = im.w != net->w || im.h != net->h;
            image sized = resize ? letterbox_image(im, net->w, net->h) : im;
            // image sized = resize_image(im, net->w, net->h);
            // image sized2 = resize_max(im, net->w);
            // image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
            // resize_network(net, sized.w, sized.h);
            layer l = net->layers[net->n - 1];

            float *X = sized.data;
            time = what_time_is_it_now();
            if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
            {
                network_predict(net, X);
            }
            if (l.type == YOLO4)
            {
                network_predict(net, X);
            }
            printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now() - time);
            int nboxes = 0;
            detection *dets = 0;
            if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
            {
                dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
            }
            if (l.type == YOLO4)
            {
                dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
            }
            // printf("%d\n", nboxes);
            if (nms)
            {
                if (l.nms_kind == DEFAULT_NMS)
                    do_nms_sort(dets, nboxes, l.classes, nms);
                else
                    diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
            {
                draw_ddetections(im, dets, nboxes, thresh, names, alphabet, l.classes, 0, 1, ffoname, margin);
            }
            if (l.type == YOLO4)
            {
                draw_ddetections(im, dets, nboxes, thresh, names, alphabet, l.classes, 0, 1, ffoname, margin);
            }
            free_detections(dets, nboxes);
            free_image(im);
            if (resize)
                free_image(sized);
            // if (filename) break;
            remove(input);
        }
        closedir(dr);
    }
}
#else
static char *lin_in_dir;
static float lin_thresh;
static float lin_hier_thresh;
static char *lin_out_dir;
static int lin_margin;
char **lin_names;
image **lin_alphabet;
network *lin_net;
float lin_nms;

int process_file(const char *file_name)
{
    // printf("fn: %s\n", file_name);

    double time = 0;

    char fname[1024];
    char ffiname[1024];
    char ffoname[1024];

    strcpy(fname, file_name);

    strcpy(ffiname, lin_in_dir);
    strcat(ffiname, "/");
    strcat(ffiname, fname);
    // printf("fi: %s\n", ffiname);

    strcpy(ffoname, lin_out_dir);
    strcat(ffoname, "/");
    strcat(ffoname, fname);
    ffoname[strlen(ffoname) - 4] = '\0';
    // printf("fo: %s\n", ffoname);

    image im = load_image_color(ffiname, 0, 0);
    if (im.w == 0 || im.h == 0)
    {
        remove(ffiname);
        return 1;
    }

    int resize = im.w != lin_net->w || im.h != lin_net->h;
    image sized = resize ? letterbox_image(im, lin_net->w, lin_net->h) : im;
    layer l = lin_net->layers[lin_net->n - 1];

    float *X = sized.data;
    time = what_time_is_it_now();
    if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
    {
        network_predict(lin_net, X);
    }
    if (l.type == YOLO4)
    {
        network_predict(lin_net, X);
    }

    int nboxes = 0;
    detection *dets = 0;
    if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
    {
        dets = get_network_boxes(lin_net, im.w, im.h, lin_thresh, lin_hier_thresh, 0, 1, &nboxes);
    }
    if (l.type == YOLO4)
    {
        dets = get_network_boxes(lin_net, im.w, im.h, lin_thresh, lin_hier_thresh, 0, 1, &nboxes);
    }
    if (lin_nms)
    {
        if (l.nms_kind == DEFAULT_NMS)
            do_nms_sort(dets, nboxes, l.classes, lin_nms);
        else
            diounms_sort_y4(dets, nboxes, l.classes, lin_nms, l.nms_kind, l.beta_nms);
    }
    if (l.type == DETECTION || l.type == REGION || l.type == YOLO)
    {
        draw_ddetections(im, dets, nboxes, lin_thresh, lin_names, lin_alphabet, l.classes, 0, 1, ffoname, lin_margin);
    }
    if (l.type == YOLO4)
    {
        draw_ddetections(im, dets, nboxes, lin_thresh, lin_names, lin_alphabet, l.classes, 0, 1, ffoname, lin_margin);
    }

    free_detections(dets, nboxes);
    free_image(im);
    if (resize)
        free_image(sized);

    remove(ffiname);

    printf("%s: Predicted in %f seconds.\n", fname, what_time_is_it_now() - time);

    return 0;
}

void test_ddetector(char *datacfg, char *cfgfile, char *weightfile, char *in_dir, float thresh, float hier_thresh, char *out_dir, int margin)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    float nms = .45;

    lin_names = names;
    lin_alphabet = alphabet;
    lin_net = net;
    lin_nms = nms;
    lin_in_dir = in_dir;
    lin_thresh = thresh;
    lin_hier_thresh = hier_thresh;
    lin_out_dir = out_dir;
    lin_margin = margin;

    while (!init_notified_file_name(in_dir, process_file))
        ;
}
#endif

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL);
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream_cv(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream_cv(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL);
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream_cv(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream_cv(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) {
        if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        else diounms_sort_y4(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
    }
}
*/

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list)
    {
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (gpu_list[i] == ',')
                ++ngpus;
        }
        gpus = (int *)calloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i)
        {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else
    {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    // int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test"))
        test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if (0 == strcmp(argv[2], "train"))
        train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if (0 == strcmp(argv[2], "valid"))
        // validate_detector(datacfg, cfg, weights, outfile);
        // validate_detector_modify————cpu+gpu协同
        validate_detector_modify(datacfg, cfg, weights, outfile);
        //validate_detector_async——批处理
        // validate_detector_async(datacfg, cfg, weights, outfile);
    else if (0 == strcmp(argv[2], "valid2"))
        validate_detector_flip(datacfg, cfg, weights, outfile);
    else if (0 == strcmp(argv[2], "recall"))
        validate_detector_recall(datacfg, cfg, weights);
    else if (0 == strcmp(argv[2], "demo"))
    {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    // else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    // else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
#undef class
