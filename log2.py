import socket

def receiver():
    PORT = 57027
    CHUNK_SIZE = 1024

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', PORT))
    s.listen(1)
    conn,address=s.accept()  # accept an incoming connection using accept() method which will block until a new client connects

    while True:
        datachunk = conn.recv(CHUNK_SIZE) # reads data chunk from the socket in batches using method recv() until it returns an empty string
        if not datachunk:
            break  # no more data coming in, so break out of the while loop
        print(datachunk)
        data.append(datachunk)  # add chunk to your already collected data

    conn.close()
    print(data)
    return

receiver()
