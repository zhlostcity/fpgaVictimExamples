/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#define MAX_SIZE 16
#define IDLE 8
#define BUSY 1800

#define SIGNAL_SIZE (1024 * 1024)

__kernel void comm_kernel(__global int4 *dst, int NUM_ACCESS) {
    int id = get_global_id(0);
    for(long i = 0; i < NUM_ACCESS; i ++) {
        dst[id] = (int)dst ^ dst[id];
    }
}

__kernel void noisegen(__global int *dst, int idle, int busy) {
    while(1) {
        for(int j = 0; j < busy; j++) {
            *dst = *dst ^ (*dst & busy) ^ idle;
        }
        int temp;
        for(int j = 0; j < idle; j++) {
            temp ++;    // dump operation
        }
    }
}