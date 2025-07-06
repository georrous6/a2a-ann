#ifndef QUEUE_H
#define QUEUE_H
#include <stddef.h>


/**
 * The nodes of the queue
 */
typedef struct QueueNode {
    void *data;
    struct QueueNode *next;
} QueueNode;

/**
 * Basic Queue implementation used by thread pools
 */
typedef struct Queue {
    QueueNode *front;
    QueueNode *rear;
    size_t dataSize;
    int n_elements;
} Queue;

/**
 * Initialize the queue with the size of the data type
 * 
 * @param q the queue to initialize
 * @param dataSize the size of the data
 */
void Queue_init(Queue *q, size_t dataSize);

/**
 * Check if the queue is empty
 * 
 * @param q the queue
 * @return 1 if the queue is empty and 0 otherwise
 */
int Queue_isEmpty(Queue *q);

/**
 * Enqueue an element to the queue
 * 
 * @param q the queue
 * @param element the element to enqueue
 * @return 1 on success and 0 if an error occured
 */
int Queue_enqueue(Queue *q, const void *element);
 

/**
 * Dequeue an element from the queue
 * 
 * @param q the queue
 * @param element the element to add
 * @return 1 on success and 0 if the queue is empty
 */
int Queue_dequeue(Queue *q, void *element);


/** 
 * Free all nodes in the queue
 * 
 * @param q the queue to be destroyed
 */
void Queue_destroy(Queue *q);

#endif
