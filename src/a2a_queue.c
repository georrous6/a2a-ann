#include "a2a_queue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void a2a_QueueInit(a2a_Queue *q, size_t dataSize) 
{
    q->front = q->rear = NULL;
    q->dataSize = dataSize;
    q->n_elements = 0;
}


int a2a_QueueIsEmpty(a2a_Queue *q) 
{
    return q->n_elements == 0;
}


int a2a_QueueEnqueue(a2a_Queue *q, const void *element) 
{
    a2a_QueueNode *newNode = (a2a_QueueNode *)malloc(sizeof(a2a_QueueNode));
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

    if (a2a_QueueIsEmpty(q)) 
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


int a2a_QueueDequeue(a2a_Queue *q, void *element) 
{
    if (a2a_QueueIsEmpty(q)) 
    {
        fprintf(stderr, "Queue is empty\n");
        return 0;
    }
    a2a_QueueNode *temp = q->front;
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
void a2a_QueueDestroy(a2a_Queue *q) 
{
    a2a_QueueNode *current = q->front;
    while (current != NULL) 
    {
        a2a_QueueNode *temp = current;
        current = current->next;
        free(temp->data);
        free(temp);
    }
    q->front = q->rear = NULL;
    q->dataSize = 0;
    q->n_elements = 0;
}
