

method = {}
def method_init():
    method[1] = 0
    method[2] = 0
    method[4] = 0
    method[8] = 0
    method[16] = 0
    method[32] = 0

def edit_method():
    method[1] = method[1] / 10
    method[2] = method[2] / 10
    method[4] = method[4] / 10
    method[8] = method[8] / 10
    method[16] = method[16] / 10

def write_average(s):
    with open(s,"a") as f:
                    f.write("1 " + str(method[1]) + "\n")
                    f.write("2 " + str(method[2]) + "\n")
                    f.write("4 " + str(method[4]) + "\n")
                    f.write("8 " + str(method[8]) + "\n")
                    f.write("16 " + str(method[16]) + "\n")


with open("results.txt","r") as fp:
    for line in fp:
        x = line.strip("\n")
        
        if x == "method 1":
            method_init()
        
        elif x == "method 2":
            edit_method()
            write_average("result-average1.txt")
            method_init()
        
        elif x == "method 3":
            edit_method()
            write_average("result-average2.txt")
            method_init()
        
        elif x == "method 4":
            edit_method()
            write_average("result-average3.txt")
            method_init()
        
        elif x == "method 5":
            edit_method()
            write_average("result-average4.txt")
            method_init()

        else:
            y = x.split(" ")
            thread = int(y[0])
            time = float(y[1])
            method[thread] += time

edit_method()
write_average("result-average5.txt")
        
#-----------------------------------------------------------------------------------------------------

def edit_chunk():
    method[1] = method[1] / 10
    method[2] = method[2] / 10
    method[4] = method[4] / 10
    method[8] = method[8] / 10
    method[16] = method[16] / 10
    method[32] = method[32] / 10

def write_average_chunk(s):
    with open(s,"a") as f1:
                    f1.write("1 " + str(method[1]) + "\n")
                    f1.write("2 " + str(method[2]) + "\n")
                    f1.write("4 " + str(method[4]) + "\n")
                    f1.write("8 " + str(method[8]) + "\n")
                    f1.write("16 " + str(method[16]) + "\n")
                    f1.write("32 " + str(method[32]) + "\n")

with open("results2.txt","r") as fp1:
    for line in fp1:
        x = line.strip("\n")
        if x == "thread 1":
            method_init()
        
        elif x == "thread 2":
            edit_chunk()
            write_average_chunk("result2-average1.txt")
            method_init()
        
        elif x == "thread 4":
            edit_chunk()
            write_average("result2-average2.txt")
            method_init()
        
        elif x == "thread 8":
            edit_chunk()
            write_average_chunk("result2-average3.txt")
            method_init()
        
        elif x == "thread 16":
            edit_method()
            write_average("result2-average4.txt")
            method_init()

        elif x == "thread 32":
            edit_method()
            write_average("result2-average5.txt")
            method_init()

        else:
            y = x.split(" ")
            thread = int(y[0])
            time = float(y[1])
            method[thread] += time

edit_chunk()
write_average_chunk("result2-average6.txt")