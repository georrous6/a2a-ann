#include "Queue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void Queue_init(Queue *q, size_t dataSize) 
{
    q->front = q->rear = NULL;
    q->dataSize = dataSize;
    q->n_elements = 0;
}


int Queue_isEmpty(Queue *q) 
{
    return q->n_elements == 0;
}


int Queue_enqueue(Queue *q, const void *element) 
{
    QueueNode *newNode = (QueueNode *)malloc(sizeof(QueueNode));
    if (!newNode)
    {
        fprintf(stderr, "Cannot add element to the queue\n");
        return 0;
    }
    newNode->data = malloc(q->dataSize);
    if (!newNode->data)
    {
        fprintf(stderr, "Cannot add element to the queue\n");
        free(newNode);
        return 0;
    }
    memcpy(newNode->data, element, q->dataSize);
    newNode->next = NULL;

    if (Queue_isEmpty(q)) 
    {
        q->front = q->rear = newNode;
    } else 
    {
        q->rear->next = newNode;
        q->rear = newNode;
    }
    q->n_elements++;
    return 1;
}


int Queue_dequeue(Queue *q, void *element) 
{
    if (Queue_isEmpty(q)) 
    {
        fprintf(stderr, "Queue is empty\n");
        return 0;
    }
    QueueNode *temp = q->front;
    memcpy(element, temp->data, q->dataSize);

    q->front = q->front->next;
    if (q->front == NULL)
    {
        q->rear = NULL;
    }

    free(temp->data);
    free(temp);
    q->n_elements--;
    return 1;
}


// Free all nodes in the queue
void Queue_destroy(Queue *q) 
{
    QueueNode *current = q->front;
    while (current != NULL) 
    {
        QueueNode *temp = current;
        current = current->next;
        free(temp->data);
        free(temp);
    }
    q->front = q->rear = NULL;
    q->dataSize = 0;
    q->n_elements = 0;
}
