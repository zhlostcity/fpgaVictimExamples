#define BUFFER_SIZE 256
#define DATA_SIZE 1024


// __kernel void comm_kernel(__global int4 *dst, int NUM_ACCESS) {
//     int id = get_global_id(0);
//     for(long i = 0; i < NUM_ACCESS; i ++) {
//         dst[id] = (int)dst ^ dst[id];
//     }
// }

__kernel void comm_kernel(__global int4 *dst, int NUM_ACCESS) {
    int id = get_global_id(0);
    int4 local_value = dst[id];
    float4 sin_input;

    // 进行sin操作
    for(long i = 0; i < NUM_ACCESS; i ++) {
        // 将int4转换为float4以便进行sin操作
        sin_input = convert_float4(local_value);
        
        // 计算正弦值
        sin_input = sin(sin_input);
        
        // 将计算结果转换回int4
        local_value = convert_int4(sin_input);
        
        // 更新内存中的值
        dst[id] = local_value;
    }
}



__kernel void convolute(
        const int IMAGE_W, const int IMAGE_H, const int HALF_FILTER_SIZE,
	const __global float * input, 
	__global float * output,
	__global float * filter 
)
{
	while(1) {

		int rowOffset = get_global_id(1) * IMAGE_W * 4;
		int my = 4 * get_global_id(0) + rowOffset;

		if (
			get_global_id(0) < HALF_FILTER_SIZE || 
			get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1 || 
			get_global_id(1) < HALF_FILTER_SIZE ||
			get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
		)
		{
			/*
			output[my] = 0.0;
			output[my+1] = 255.0;
			output[my+2] = 255.0;
			output[my+3] = 255.0;
			*/
		
			return;
		}
	
		else
		{
			// perform convolution
			int fIndex = 0;
			output[my] = 0.0;
			output[my+1] = 0.0;
			output[my+2] = 0.0;
			output[my+3] = 0.0;
		
			for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
			{
				int curRow = my + r * (IMAGE_W * 4);
				for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
				{
					int offset = c * 4;
				
					output[ my   ] += input[ curRow + offset   ] * filter[ fIndex   ]; 
					output[ my+1 ] += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
					output[ my+2 ] += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
					output[ my+3 ] += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
				
					fIndex += 4;
	
				}	
			}	
		}
	}
}
