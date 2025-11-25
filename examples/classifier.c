#include <jiaoben.h>
#include <pthread.h>
#include "darknet.h"
#include "image.h"
#ifdef WIN32
#include "utils.h"
#endif

#ifdef WIN32
#include "unistd\sys\time.h"
#else
#include <sys/time.h>
#endif

#include <assert.h>
#define class temp

float *get_regression_values(char **labels, int n)
{
    float *v = (float*)calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = (network**)calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        if(gpu_index >= 0){
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
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen)/N;

    if(count == 0) {
        char buff[256];
        sprintf(buff, "%s/%s.start.conv.weights",backup_directory,base);
        save_weights(net, buff);
    }

#ifdef LOSS_ONLY
    time = what_time_is_it_now();
#endif
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("Resizing\n");
#endif
            int dim = (rand() % 11 + 4) * 32;
            //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("%d\n", dim);
#endif
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
#ifndef BENCHMARK
            printf("%d %d\n", args.min, args.max);
#endif
            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
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
#ifndef LOSS_ONLY
        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
#endif
#ifndef LOSS_ONLY
        time = what_time_is_it_now();
#endif
        float loss = 0;
#ifdef GPU
        if (gpu_index >= 0) {
            if (ngpus == 1) {
                loss = train_network(net, train);
            } else {
                loss = train_networks(nets, ngpus, train, 4);
            }
        }
        else {
            loss = train_network(net, train);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
#ifdef LOSS_ONLY
        printf("%lf\t%f\n", what_time_is_it_now()-time, loss);
#else
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
#endif
#ifdef GPU
        if (loss != loss && gpu_index >= 0) {
            opencl_deinit(gpusg, ngpusg);
        }
#endif
        if(loss != loss) { printf("NaN LOSS detected! No possible to continue!\n"); exit(-7); }
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
#ifdef GPU_STATS
        opencl_dump_mem_stat();
#endif
#ifdef BENCHMARK
        break;
#endif
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 2);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = (int*)calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net->w;
        int h = net->h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = (float*)calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = (int*)calloc(topk, sizeof(int));

    int size = net->w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}




// pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;  // 互斥锁

// char **paths;  // 图片路径数组
// int current_image_index = 0;  // 当前处理的图片索引
// int total_image = 32;  // 图片总数
// int all_images_processed = 0;  // 用于标识是否所有图片已处理

// typedef struct {
//     network *net;
//     int device_id;  // 设备ID，0代表CPU，1代表GPU
//     int topk;
//     int classes;
//     char **labels;
// } thread_data_t;

// // 消费者函数：从公共图片路径数组中获取图片进行推理
// void* consumer(void *arg) {
//     thread_data_t *data = (thread_data_t*)arg;
//     network *net = data->net;
//     int topk = data->topk;
//     int classes = data->classes;
//     char **labels = data->labels;

//     int *indexes = (int*)calloc(topk, sizeof(int));
//     float avg_acc = 0;
//     float avg_topk = 0;
//     int i = 0;

//     while (1) {
//         char *path = NULL;
//         int image_index = 0;
//         pthread_mutex_lock(&mutex1);
//         int remaining_images = total_image - current_image_index;  // 剩余图片数量
//         // if (data->device_id == 0 && remaining_images <= 2) {
//         if (data->device_id == 1 && remaining_images <= 1) {
//             // 如果 CPU 且剩余图片少于阈值，停止 CPU 推理
//             // printf("Stopping CPU inference as remaining images <= %d\n", 2);
//             printf("Stopping GPU inference as remaining images <= %d\n", 1);
//             pthread_mutex_unlock(&mutex1);
//             break;
//         }
//         if (current_image_index < total_image) {
//             path = paths[current_image_index];  // 获取当前图片路径
//             image_index = current_image_index;
//             current_image_index++;
//         } else {
//             all_images_processed = 1;  // 所有图片都已处理
//         }
//         pthread_mutex_unlock(&mutex1);

//         if (path == NULL && all_images_processed) break;  // 如果没有图片可处理，退出
//         opencl_device_id_t = data->device_id;
//         // 推理过程
//         double inference_start_time = what_time_is_it_now();
//         int class = -1;
//         for (int j = 0; j < classes; ++j) {
//             if (strstr(path, labels[j])) {
//                 class = j;
//                 break;
//             }
//         }

        
//         image im = load_image_color(path, 0, 0);
//         image crop = center_crop_image(im, net->w, net->h);
        
//         float *pred = network_predict(net, crop.data);
//         double inference_end_time = what_time_is_it_now();

//         free_image(im);
//         free_image(crop);
//         top_k(pred, classes, topk, indexes);

//         if (indexes[0] == class) avg_acc += 1;
//         for (int j = 0; j < topk; ++j) {
//             if (indexes[j] == class) avg_topk += 1;
//         }

//         // 输出推理时间和结果
//         printf("****** Device %d Inference time for image %d: %f seconds\n", data->device_id, image_index, inference_end_time - inference_start_time);
//         // printf("******%s, %d, %f, %f, \n", path, class, pred[0], pred[1]);
//         // printf("******%d: top 1: %f, top %d: %f\n\n", image_index, avg_acc / (i + 1), topk, avg_topk / (i + 1));
//         i++;
//     }

//     free(indexes);
//     return NULL;
// }

// void validate_classifier_dynamic_buffer(char *datacfg, char *filename, char *weightfile) {
//     // 初始化网络和配置
//     opencl_device_id_t = 0 ;
//     network *net_cpu = load_network(filename, weightfile, 0);
//     opencl_device_id_t = 1;
//     network *net_gpu = load_network(filename, weightfile, 0);
//     set_batch_network(net_cpu, 1);
//     set_batch_network(net_gpu, 1);
//     srand(time(0));

//     list *options = read_data_cfg(datacfg);
//     char *label_list = option_find_str(options, "labels", "data/labels.list");
//     char *leaf_list = option_find_str(options, "leaves", 0);
//     if (leaf_list) change_leaves(net_cpu->hierarchy, leaf_list);
//     char *valid_list = option_find_str(options, "valid", "data/train.list");
//     int classes = option_find_int(options, "classes", 2);
//     int topk = option_find_int(options, "top", 1);
//     char **labels = get_labels(label_list);
//     list *plist = get_paths(valid_list);
//     paths = (char **)list_to_array(plist);
//     // total_image = plist->size;
//     free_list(plist);


//     double start = what_time_is_it_now();
//     // 创建两个线程，一个用于CPU推理，一个用于GPU推理
//     pthread_t thread_cpu, thread_gpu;
//     thread_data_t data_cpu = {net_cpu, 0, topk, classes, labels};
//     thread_data_t data_gpu = {net_gpu, 1, topk, classes, labels};

//     pthread_create(&thread_cpu, NULL, consumer, &data_cpu);
//     pthread_create(&thread_gpu, NULL, consumer, &data_gpu);

//     // 等待两个线程结束
//     pthread_join(thread_cpu, NULL);
//     pthread_join(thread_gpu, NULL);

//     fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
//     // 释放资源
//     // free_network(net_cpu);
//     // free_network(net_gpu);
//     free(paths);
// }


void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    opencl_device_id_t = 1 ;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    double start = what_time_is_it_now();
    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = (int*)calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        double inference_start_time = what_time_is_it_now();
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        //grayscale_image_3c(crop);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);
        double inference_end_time = what_time_is_it_now();
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }
        
        printf("******Inference time for image %d: %f seconds\n",  i , inference_end_time-inference_start_time);
        
        // printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        // printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_classifier_multi(char *datacfg, char *cfg, char *weights)
{
    int i, j;
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    //int scales[] = {224, 288, 320, 352, 384};
    int scales[] = {224, 256, 288, 320};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = (int*)calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = (float*)calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_max(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = (int*)calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net->layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        if(gpu_index >= 0) {
            opencl_pull_array(l.output_gpu, l.output, l.outputs);
        }
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{   
    opencl_device_id_t = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = (int*)calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        //Original
        // int resize = im.w != net->w || im.h != net->h;
        // image r = resize ? letterbox_image(im, net->w, net->h) : im;
        
        
		//For MobileNet v2
		image r = resize_image(im, net->w, net->h);

        //image r = resize_min(im, 320);
        //printf("%d %d\n", r.w, r.h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        double start = what_time_is_it_now();
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-start);
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;// 互斥锁

char **paths;           // 图片路径数组
int current_image_index = 0;  // 当前处理的图片索引
int total_image = 60;    // 图片总数
int all_images_processed = 0;  // 用于标识是否所有图片已处理
#define SPEED_ESTIMATE_BATCH 2 // 第几个批次用于速度估算
#define TAIL_THRESHOLD_RATIO 0.15 // 最后10%图片开始动态分配
int gpu_count=0;
typedef struct {
    float avg_speed;
} thread_stat_t;
thread_stat_t thread_stats[2]; // 支持 CPU 和 GPU

thread_stat_t thread_stats[2]; // 支持 CPU、GPU 的速度记录
typedef struct {
    network *net;  
    int device_id;  // 设备ID，0代表CPU，1代表GPU
    int classes;    
    char **labels; 
    int batch_counter; // 已处理批次数
    
} thread_data_t;
// typedef struct {
//     network *net;  
//     int device_id;  // 设备ID，0代表CPU，1代表GPU
//     int batch_size; 
//     int classes;    
//     char **labels;  
// } thread_data_t;
void* consumer_async2(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;
    network *net = data->net;
    int classes = data->classes;
    char **labels = data->labels;
    int base_batch_size = (data->device_id == 0) ? CPU_BATCH : GPU_BATCH;
    int batch_size = base_batch_size;

    while (1) {
        char **batch_paths = malloc(batch_size * sizeof(char*));
        int batch_count = 0;

        pthread_mutex_lock(&mutex1);
        int remaining_images = total_image - current_image_index;

        // 动态调整分配策略
        if (remaining_images <= total_image * 0.15) {
            int total_remain = total_image - current_image_index;
            float cpu_time = thread_stats[0].avg_speed > 0 ? thread_stats[0].avg_speed : 1.0f;
            float gpu_time = thread_stats[1].avg_speed > 0 ? thread_stats[1].avg_speed : 0.3f;

            int cpu_alloc = (int)((gpu_time / (cpu_time + gpu_time)) * total_remain);
            int gpu_alloc = total_remain - cpu_alloc;

            // // 判断该线程是否继续处理
            // if ((data->device_id == 0 && cpu_alloc == 0) || (data->device_id == 1 && gpu_alloc == 0)) {
            //     pthread_mutex_unlock(&mutex1);
            //     free(batch_paths);
            //     break;
            // }

            // 动态调整batch_size
            batch_size = (data->device_id == 0) ? (cpu_alloc < base_batch_size ? cpu_alloc : base_batch_size)
                                                : (gpu_alloc < base_batch_size ? gpu_alloc : base_batch_size);
            // if (data->device_id == 0 && remaining_images <= 6) {batch_size=1;}
            // if (batch_size < 1) batch_size = 1;
            set_batch_network(net, batch_size);
        }
        //  if (data->device_id == 0 && remaining_images <= 2) {
        //     // 如果 CPU 且剩余图片少于阈值，停止 CPU 推理
        //     printf("Stopping CPU inference as remaining images <= %d\n", 2);
        //     pthread_mutex_unlock(&mutex1);
        //     break;
        // }

        // 拿图
        for (int i = 0; i < batch_size && current_image_index < total_image; i++) {
            batch_paths[i] = paths[current_image_index++];
            batch_count++;
        }
        if (current_image_index >= total_image) all_images_processed = 1;
        pthread_mutex_unlock(&mutex1);

        if (batch_count == 0) {
            free(batch_paths);
            if (all_images_processed) return;
            continue;
        }

        if (batch_count < batch_size) {
            batch_size = batch_count;
            set_batch_network(net, batch_size);
        }

        opencl_device_id_t = data->device_id;
        double batch_start_time = what_time_is_it_now();

        int input_size = net->w * net->h * net->c;
        float *batch_data = calloc(batch_size * input_size, sizeof(float));

        for (int i = 0; i < batch_size; i++) {
            image im = load_image_color(batch_paths[i], 0, 0);
            image resized = resize_min(im, net->w);
            image crop = crop_image(resized, (resized.w - net->w) / 2, (resized.h - net->h) / 2, net->w, net->h);
            memcpy(batch_data + i * input_size, crop.data, input_size * sizeof(float));
            if (resized.data != im.data) free_image(resized);
            free_image(im);
            free_image(crop);
        }

        float *predictions = network_predict(net, batch_data);
        for (int i = 0; i < batch_size; i++) {
            float *pred = predictions + i * classes;
            int ind = max_index(pred, classes);
            printf("Device %d: Image %s predicted as %s\n", data->device_id, batch_paths[i], labels[ind]);
        }

        double batch_end_time = what_time_is_it_now();
        double elapsed = batch_end_time - batch_start_time;
        printf("Device %d: Batch time: %.4f s\n", data->device_id, elapsed);

        // 👇 第三批推理后记录平均速度
        data->batch_counter++;
        if (data->batch_counter == 2 && thread_stats[data->device_id].avg_speed == 0) {
            thread_stats[data->device_id].avg_speed = elapsed / batch_size;
            printf("Device %d: Estimated avg speed per image = %.6f s\n", data->device_id, thread_stats[data->device_id].avg_speed);
        }

        free(batch_data);
        free(batch_paths);
    }

    return NULL;
}


void label_classifier_async(char *datacfg, char *filename, char *weightfile) {
    thread_stats[0].avg_speed = 0;
    thread_stats[1].avg_speed = 0;
    // 初始化网络和配置
    opencl_device_id_t = 0;
    network *net_cpu = load_network(filename, weightfile, 0);
    opencl_device_id_t = 1;
    network *net_gpu1 = load_network(filename, weightfile, 0);
    // 为两个网络设置批处理大小
    set_batch_network(net_cpu, CPU_BATCH);
    set_batch_network(net_gpu1, GPU_BATCH);
    
    srand(time(0));
    // 读取参数
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    // 加载标签和图片地址
    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);
    paths = (char **)list_to_array(plist);
    // total_image = plist->size;  // 设置图片总数
    free_list(plist);
    //记录开始时间
    double start = what_time_is_it_now();
    // 创建两个线程，一个用于CPU推理，一个用于GPU推理
    pthread_t thread_cpu, thread_gpu1;
    
    thread_data_t cpu_data = {net_cpu, 0, classes, labels,.batch_counter=0};
    thread_data_t gpu1_data = {net_gpu1, 1, classes, labels,.batch_counter=0};

    pthread_create(&thread_cpu, NULL, consumer_async2, &cpu_data);
    pthread_create(&thread_gpu1, NULL, consumer_async2, &gpu1_data);
   
    // 等待两个线程结束
    pthread_join(thread_cpu, NULL);
    pthread_join(thread_gpu1, NULL);
  
    // 记录结束时间
    fprintf(stderr, "Total Inference Time: %f Seconds\n", what_time_is_it_now() - start);
    // printf("******GPU inference total images: %d  *****\n", gpu_count);
    // 释放资源
    // free(paths);
}

// 消费者函数：从公共图片路径数组中获取图片进行推理
void* consumer_async(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;
    network *net = data->net;
    int classes = data->classes;
    char **labels = data->labels;
    int batch_size = (data->device_id == 0) ? CPU_BATCH : GPU_BATCH;

    while (1) {
        char **batch_paths = malloc(batch_size * sizeof(char*));
        int batch_count = 0;
        pthread_mutex_lock(&mutex1);
        
        int remaining_images = total_image - current_image_index;  // 剩余图片数量
        
        // if(remaining_images<15 && batch_size!=1){batch_size = 1;
        // set_batch_network(net, batch_size);
        // }
        // if(data->device_id == 1 && remaining_images<=12 && batch_size!=4){batch_size = 4;
        // set_batch_network(net, batch_size);
        // }
        // if(data->device_id == 1 && remaining_images<=6 && batch_size!=1){batch_size = 1;
        // set_batch_network(net, batch_size);
        // }
        // if (data->device_id == 0 && remaining_images <= 2) {
        //     // 如果 CPU 且剩余图片少于阈值，停止 CPU 推理
        //     printf("Stopping CPU inference as remaining images <= %d\n", 2);
        //     pthread_mutex_unlock(&mutex1);
        //     break;
        // }


        // 从任务队列中分配一个批次的图片路径
        for (int i = 0; i < batch_size && current_image_index < total_image; i++) {
            batch_paths[i] = paths[current_image_index];
            current_image_index++;
            batch_count++;
        }
        if (current_image_index >= total_image) {
            all_images_processed = 1; // 标记所有图片处理完
        }
        pthread_mutex_unlock(&mutex1);

        // 如果没有图片可处理，退出
        if (batch_count == 0) {
            free(batch_paths);
            if (all_images_processed) break;
            continue;
        }
        //GPU最后图片不足，重新设置批大小
        if (batch_count<batch_size){set_batch_network(net, batch_count);}

        // 设置当前设备
        opencl_device_id_t = data->device_id;
        double batch_start_time = what_time_is_it_now();

        // 准备批次数据：合并 batch_count 张图片的数据到一个大的输入张量
        int input_size = net->w * net->h * net->c;  // 网络输入张量的单张图片大小
        float *batch_data = calloc(batch_count * input_size, sizeof(float)); // 创建批次数据

        for (int i = 0; i < batch_count; i++) {
            image im = load_image_color(batch_paths[i], 0, 0);
            image resized = resize_min(im, net->w);
            image crop = crop_image(resized, (resized.w - net->w) / 2, (resized.h - net->h) / 2, net->w, net->h);

            memcpy(batch_data + i * input_size, crop.data, input_size * sizeof(float)); // 将数据合并到批次张量中

            // 释放图片内存
            if (resized.data != im.data) free_image(resized);
            free_image(im);
            free_image(crop);
        }

        // 批量推理：一次性处理整个批次的数据
        float *predictions = network_predict(net, batch_data);
        
        // 输出结果
        for (int i = 0; i < batch_count; i++) {
            float *pred = predictions + i * classes; // 每张图片的预测结果
            int ind = max_index(pred, classes);
            printf("Device %d: Image %s predicted as %s\n", data->device_id, batch_paths[i], labels[ind]);
        }

        double batch_end_time = what_time_is_it_now();
        printf("Device %d: Batch inference time: %f seconds\n", data->device_id, batch_end_time - batch_start_time);

        // 释放内存
        free(batch_data);
        free(batch_paths);
    }
    return NULL;
}

// 消费者函数：从公共图片路径数组中获取图片进行推理
void* consumer(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;
    network *net = data->net;
    int classes = data->classes;
    char **labels = data->labels;

    while (1) {
        char *path = NULL;
        int image_index = 0;
        pthread_mutex_lock(&mutex1);
        int remaining_images = total_image - current_image_index;  // 剩余图片数量
        // if ((data->device_id == 0 || data->device_id == 1 || data->device_id == 2 || data->device_id == 3) && remaining_images <= 4) {
        // if ((data->device_id != 8|| data->device_id != 9)  && remaining_images <= 4) {
        // // if ((data->device_id == 1 ||data->device_id == 0) && remaining_images <= 3) {
        //     // 如果 CPU 且剩余图片少于阈值，停止 CPU 推理
        //     printf("Stopping CPU inference as remaining images <= %d\n", 4);
        //     // printf("Stopping GPU inference as remaining images <= %d\n", 1);
        //     pthread_mutex_unlock(&mutex1);
        //     break;
        // }
        // 检查是否还有图片未处理
        if (current_image_index < total_image) {
            path = paths[current_image_index];  //获取当前图片路径
            image_index = current_image_index;
            current_image_index++;  
        } else {
            all_images_processed = 1;  // 标记所有图片处理完
        }

        pthread_mutex_unlock(&mutex1);

        // 如果没有图片可处理，退出
        if (path == NULL && all_images_processed) break;

        // 指定设备ID
        opencl_device_id_t = data->device_id;

        double inference_start_time = what_time_is_it_now();
        // 执行推理
        image im = load_image_color(path, 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w) / 2, (resized.h - net->h) / 2, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        double inference_end_time = what_time_is_it_now();

        // Free image memory
        if (resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);

        // Get the index of the predicted class获取预测类别的索引
        int ind = max_index(pred, classes);
        // gpu_count=gpu_count+(data->device_id==1);
        // 输出推理时间和结果
        printf("****** Device %d Inference time for image %d: %f seconds\n", data->device_id, image_index, inference_end_time - inference_start_time);
        // 打印预测类别的标签
        printf("Device %d inferenceimage %d: %s\n",data->device_id, image_index, labels[ind]);
    }
    return NULL;
}


void label_classifier_modify1(char *datacfg, char *filename, char *weightfile) {
    // 设置设备数量（0为CPU，其余为GPU）
    int device_count = 4; // 1个CPU + 1个GPU（根据实际需要修改，比如想启用4个GPU就设为5）
    int batch_sizes[4] = {1,1,1,1};

    // 初始化网络数组和线程数组
    network *nets[device_count];
    pthread_t threads[device_count];
    thread_data_t thread_data[device_count];

    // 加载网络并设置 batch 大小
    for (int i = 0; i < device_count; ++i) {
        opencl_device_id_t = i;
        nets[i] = load_network(filename, weightfile, 0);
        set_batch_network(nets[i], batch_sizes[i]);
    }

    srand(time(0));

    // 读取参数
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    // 加载标签和图片地址
    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);
    paths = (char **)list_to_array(plist);
    free_list(plist);

    // 记录开始时间
    double start = what_time_is_it_now();

    // 创建多个推理线程
    for (int i = 0; i < device_count; ++i) {
        thread_data[i].net = nets[i];
        thread_data[i].device_id = i;
        thread_data[i].classes = classes;
        thread_data[i].labels = labels;
        pthread_create(&threads[i], NULL, consumer, &thread_data[i]);
    }

    // 等待线程结束
    for (int i = 0; i < device_count; ++i) {
        pthread_join(threads[i], NULL);
    }

    // 输出总用时
    fprintf(stderr, "Total Inference Time: %f Seconds\n", what_time_is_it_now() - start);

    // 后续可以释放 nets[i] 和 paths 等资源
}

void label_classifier_modify(char *datacfg, char *filename, char *weightfile) {
    // 初始化网络和配置
    // opencl_device_id_t = 0;
    // network *net_cpu = load_network(filename, weightfile, 0);
    opencl_device_id_t = 1;
    network *net_gpu1 = load_network(filename, weightfile, 0);
    // opencl_device_id_t = 2;
    // network *net_gpu2 = load_network(filename, weightfile, 0);
    // opencl_device_id_t = 3;
    // network *net_gpu3 = load_network(filename, weightfile, 0);
    // opencl_device_id_t = 4;
    // network *net_gpu4 = load_network(filename, weightfile, 0);
    // 为两个网络设置批处理大小
    // set_batch_network(net_cpu, CPU_BATCH);
    set_batch_network(net_gpu1, GPU_BATCH);
    // set_batch_network(net_gpu2, GPU_BATCH);
    // set_batch_network(net_gpu3, GPU_BATCH);
    // set_batch_network(net_gpu4, 1);
    srand(time(0));
    // 读取参数
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    // 加载标签和图片地址
    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);
    paths = (char **)list_to_array(plist);
    // total_image = plist->size;  // 设置图片总数
    free_list(plist);
    //记录开始时间
    double start = what_time_is_it_now();
    // 创建两个线程，一个用于CPU推理，一个用于GPU推理
    pthread_t thread_cpu, thread_gpu1, thread_gpu2, thread_gpu3, thread_gpu4;
    // thread_data_t data_cpu = {net_cpu, 0, classes, labels};
    thread_data_t data_gpu1 = {net_gpu1, 1, classes, labels};
    // thread_data_t data_gpu2 = {net_gpu2, 2, classes, labels};
    // thread_data_t data_gpu3 = {net_gpu3, 3, classes, labels};
    // thread_data_t data_gpu4 = {net_gpu4, 4, classes, labels};
    // pthread_create(&thread_cpu, NULL, consumer, &data_cpu);
    pthread_create(&thread_gpu1, NULL, consumer, &data_gpu1);
    // pthread_create(&thread_gpu2, NULL, consumer, &data_gpu2);
    // pthread_create(&thread_gpu3, NULL, consumer, &data_gpu3);
    // pthread_create(&thread_gpu4, NULL, consumer, &data_gpu4);
    // 等待两个线程结束
    // pthread_join(thread_cpu, NULL);
    pthread_join(thread_gpu1, NULL);
    // pthread_join(thread_gpu2, NULL);
    // pthread_join(thread_gpu3, NULL);
    // pthread_join(thread_gpu4, NULL);
    // 记录结束时间
    fprintf(stderr, "Total Inference Time: %f Seconds\n", what_time_is_it_now() - start);
    // printf("******GPU inference total images: %d  *****\n", gpu_count);
    // 释放资源
 
    // free(paths);
}

void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    opencl_device_id_t = 0;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    // int m = plist->size;
    int m = 52;
    free_list(plist);
    //记录开始时间
    double start = what_time_is_it_now();
    for(i = 0; i < m; ++i){
        double inference_start_time = what_time_is_it_now();
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        double inference_end_time = what_time_is_it_now();
        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);
        printf("******Inference time for image %d: %f seconds\n",  i , inference_end_time-inference_start_time);
        // printf("%s\n", labels[ind]);
    }
    // 记录结束时间
    fprintf(stderr, "Total Inference Time: %f Seconds\n", what_time_is_it_now() - start);

}

void csv_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    int i,j;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int top = option_find_int(options, "top", 1);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    int *indexes = (int*)calloc(top, sizeof(int));

    for(i = 0; i < m; ++i){
        double time = what_time_is_it_now();
        char *path = paths[i];
        image im = load_image_color(path, 0, 0);
        int resize = im.w != net->w || im.h != net->h;
        image r = resize ? letterbox_image(im, net->w, net->h) : im;
        float *predictions = network_predict(net, r.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);

        printf("%s", path);
        for(j = 0; j < top; ++j){
            printf("\t%d", indexes[j]);
        }
        printf("\n");

        free_image(im);
        if (resize) free_image(r);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", what_time_is_it_now() - time, i+1, m);
    }
}

void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
    int curr = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = classes;
    args.n = net->batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net->batch; curr < m; curr += net->batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net->batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;
        if (target_layer >= 0){
            //layer l = net->layers[target_layer];
        }

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net->batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}

void file_output_classifier(char *datacfg, char *filename, char *weightfile, char *listfile)
{
    int i,j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    //char *label_list = option_find_str(options, "names", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(listfile);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);

        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 0, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);

        printf("%s", paths[i]);
        for(j = 0; j < classes; ++j){
            printf("\t%g", pred[j]);
        }
        printf("\n");
    }
}


void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = (int*)calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    //cvNamedWindow("Threat", CV_WINDOW_NORMAL); 
    //cvResizeWindow("Threat", 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream_cv(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net->w, net->h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = 0;
        if(1){
            curr_threat = predictions[0] * 0 + 
                predictions[1] * .6 + 
                predictions[2];
        } else {
            curr_threat = predictions[218] +
                predictions[539] + 
                predictions[540] + 
                predictions[368] + 
                predictions[369] + 
                predictions[370];
        }
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "/home/piotr/tmp/threat_%06d", count);
        //save_image(out, buff);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(1){
            show_image(out, "Threat", 10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = (int*)calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream_cv(cap);
        image in_s = resize_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");

        int threat = 0;
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("Threat Detected!\n");
                threat = 1;
                break;
            }
        }
        if(!threat) printf("Scanning...\n");
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("%s\n", names[index]);
            }
        }

        show_image(in, "Threat Detection", 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    image **alphabet = load_alphabet();
    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);

    int w = 1280;
    int h = 720;
    void * cap = open_video_stream(filename, cam_index, w, h, 0);

    int top = option_find_int(options, "top", 1);

    char *label_list = option_find_str(options, "labels", 0);
    char *name_list = option_find_str(options, "names", label_list);
    char **names = get_labels(name_list);

    int *indexes = (int*)calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream_cv(cap);
        //image in_s = resize_image(in, net->w, net->h);
        int resize = in.w != net->w || in.h != net->h;
        image in_s = resize ? letterbox_image(in, net->w, net->h) : in;

        float *predictions = network_predict(net, in_s.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        int lh = in.h*.03;
        int toph = 3*lh;

        float rgb[3] = {1,1,1};
        for(i = 0; i < top; ++i){
            printf("%d\n", toph);
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);

            char buff[1024];
            sprintf(buff, "%3.1f%%: %s\n", predictions[index]*100, names[index]);
            image label = get_label(alphabet, buff, lh);
            draw_label(in, toph, lh, label, rgb);
            toph += 2*lh;
            free_image(label);
        }

        show_image(in, base, 10);
        if (resize) free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);


    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "fout")) file_output_classifier(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    else if(0==strcmp(argv[2], "csv")) csv_classifier(data, cfg, weights);
    // else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "label")) label_classifier_modify1(data, cfg, weights);
    // else if(0==strcmp(argv[2], "label")) label_classifier_async(data, cfg, weights);

    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    // else if(0==strcmp(argv[2], "valid")) validate_classifier_dynamic_buffer(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}
#undef class
