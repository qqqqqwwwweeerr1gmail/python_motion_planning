


def gen_block(start=(10,10),end= (12,12)):
    block_cell = []
    start_x = min(start[0],end[0])
    start_y = min(start[1],end[1])
    end_x = max(start[0],end[0])
    end_y = max(start[1],end[1])

    for i in range(start_x,end_x):
        block_cell.append((i,start_y))
        block_cell.append((i,end_y))

    for y in range(start_y,end_y):
        block_cell.append((start_x,y))
        block_cell.append((end_x,y))

    block_cell.append((end_x,end_y))
    return block_cell

if __name__ == '__main__':
    block_cell = gen_block(start=(10,30),end= (20,40))
    block_cell = gen_block(start=(10,30),end= (20,40))
    print(block_cell)
    print(type(block_cell))














