#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "vector.h"

#define REPEAT_TIME 10

pthread_t thread[4];

void* thread0 ( void* )
{
    // do GPU task
    float  time, start;
    start  =  clock();
    printf( " Thread 0  start\n " );
    int  i, n  =   100 ;
    float* a,  * b,  * c;
    a  =  ( float* )malloc(n  *   sizeof ( float ));
    b  =  ( float* )malloc(n  *   sizeof ( float ));
    c  =  ( float* )malloc(n  *   sizeof ( float ));
    for (i  =   0 ; i  <  n; i ++ )
    {
        a[i]  =   1.0f ;
        b[i]  =   1.0f ;
    }
    
    for (i = 0 ; i < REPEAT_TIME; i++ ) vectorAdd(a, b, c, n);
    for (i = 0 ; i < n; i ++ ) printf( " Thread 0 :c[%d] = %f\n " ,  i , c[i]);

    free(a);
    free(b);
    free(c);
    time  =  clock()  -  start;
    printf( " Thread 0 : task was finished!\ncostTime0 : %f\n " , time  /  CLOCKS_PER_SEC);
    pthread_exit(NULL);
}

void* thread1 ( void* )
{
    // do GPU task
    float  time, start;
    start  =  clock();
    printf( " Thread 1 start\n " );
    int  i, n  =   100 ;
    float* a,  * b,  * c;
    a  =  ( float* )malloc(n  *   sizeof ( float ));
    b  =  ( float* )malloc(n  *   sizeof ( float ));
    c  =  ( float* )malloc(n  *   sizeof ( float ));
    for (i  =   0 ; i < n; i ++ )
    {
        a[i]  =   1.6f ;
        b[i]  =   2.0f ;
    }

    for (i = 0 ; i < REPEAT_TIME ; i++ ) vectorMul(a, b, c, n);
    for (i = 0 ; i < n; i ++ ) printf( " Thread 1 :c[%d] = %f\n " ,  i , c[i]);
    
    free(a);
    free(b);
    free(c);
    time  =  clock()  -  start;
    printf( " Thread 1 : task was finished!\ncostTime1 : %f\n " , time  /  CLOCKS_PER_SEC);
    pthread_exit(NULL);
}

void  thread_create()
{
    int  temp;
    memset( & thread,  0 ,  sizeof (thread));
    if ((temp  =  pthread_create( & thread[ 0 ], NULL, thread0, NULL))  !=   0 )
        printf( " Thread 0 Created Failed!\n " );
    else
        printf( " Thread 0 Created!\n " );
    
    if ((temp  =  pthread_create( & thread[ 1 ], NULL, thread1, NULL))  !=   0 )
        printf( " Thread 1 Created Failed!\n " );
    else
        printf( " Thread 1 Created!\n " );
}

void  thread_wait()
{
    if (thread[ 0 ]  !=   0 )
    {
        pthread_join(thread[ 0 ], NULL);
        printf( " Thread 0 Done\n " );
    }
    if (thread[ 1 ]  !=   0 )
    {
        pthread_join(thread[ 1 ], NULL);
        printf( " Thread 1 Done\n " );
    }
}


int  main()
{
    float  time, start;
    printf( " Creating Thread \n " );
    start  =  clock();
    thread_create();
    printf( " Waiting for Thread Done\n " );
    thread_wait();
    time  =  clock()  -  start;
    printf( " Overall Cost Time : %f\n " , time  /  CLOCKS_PER_SEC);
    return 0;
}