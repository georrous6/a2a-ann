#ifndef A2A_QUEUE_H
#define A2A_QUEUE_H
#include <stddef.h>


/**
 * The nodes of the queue
 */
typedef struct a2a_QueueNode {
    void *data;
    struct a2a_QueueNode *next;
} a2a_QueueNode;

/**
 * Basic Queue implementation used by thread pools
 */
typedef struct a2a_Queue {
    a2a_QueueNode *front;
    a2a_QueueNode *rear;
    size_t dataSize;
    int n_elements;
} a2a_Queue;

/**
 * Initialize the queue with the size of the data type
 * 
 * @param q the queue to initialize
 * @param dataSize the size of the data
 */
void a2a_QueueInit(a2a_Queue *q, size_t dataSize);

/**
 * Check if the queue is empty
 * 
 * @param q the queue
 * @return 1 if the queue is empty and 0 otherwise
 */
int a2a_QueueIsEmpty(a2a_Queue *q);

/**
 * Enqueue an element to the queue
 * 
 * @param q the queue
 * @param element the element to enqueue
 * @return 1 on success and 0 if an error occured
 */
int a2a_QueueEnqueue(a2a_Queue *q, const void *element);
 

/**
 * Dequeue an element from the queue
 * 
 * @param q the queue
 * @param element the element to add
 * @return 1 on success and 0 if the queue is empty
 */
int a2a_QueueDequeue(a2a_Queue *q, void *element);


/** 
 * Free all nodes in the queue
 * 
 * @param q the queue to be destroyed
 */
void a2a_QueueDestroy(a2a_Queue *q);

#endif
