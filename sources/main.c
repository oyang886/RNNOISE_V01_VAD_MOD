#pragma comment(lib, "D:\\Program Files\\Mega-Nerd\\libsndfile\\lib\\libsndfile-1.lib")
#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<sndfile.h>
#include"vad_dect.h"
#include <time.h>

clock_t start, end;
double cpu_time_used;

#define FRAME_SIZE 480
#define HALF FRAME_SIZE

int main()
{
	float input_buffer[HALF];
	float output_buffer[HALF];
	const char* input_file = "orig_audio.wav";
	const char* output_file = "mod_audio.wav";
	float test_vad = 0.0f;
	int i;
	vad_dect_init();

	int frames_read;

	printf("开始处理\n");
	printf("输入文件: %s\n", input_file);
	printf("输出文件: %s\n", output_file);

	SF_INFO sfinfo;
	SNDFILE* sf_in = sf_open(input_file, SFM_READ, &sfinfo);
	if (!sf_in) {
		printf("错误: 无法打开输入WAV文件\n");
		printf("libsndfile错误: %s\n", sf_strerror(NULL));
		return 1;
	}
	printf("成功打开输入文件\n");

	SNDFILE* sf_out = sf_open(output_file, SFM_WRITE, &sfinfo);
	if (!sf_out) {
		printf("错误: 无法创建输出文件\n");
		printf("libsndfile错误: %s\n", sf_strerror(NULL));
		sf_close(sf_in);
		return 1;
	}
	printf("成功创建输出文件\n");

	// 开始
	start = clock();
	while ((frames_read = sf_readf_float(sf_in, input_buffer, HALF)) > 0)
	{
		for (i = frames_read; i < HALF; i++) input_buffer[i] = 0;
		for (i = 0; i < HALF; i++)
		{
			input_buffer[i] *= 32768.0f;
		}
		test_vad = rnn_get_vad(input_buffer);
		// printf("vad: %f\n", test_vad);
		if (test_vad > 0.7f)
		{
			for (i = 0; i < HALF; i++)
			{
				output_buffer[i] = input_buffer[i] * 0.000030517578125f;
			}
			sf_writef_float(sf_out, output_buffer, frames_read);
		}
	}
	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Function execution time: %f seconds\n", cpu_time_used);

	sf_close(sf_in);
	sf_close(sf_out);
	printf("完成！\n");
	return 0;
}
