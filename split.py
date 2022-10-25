

path='xxx'

path_mytrain='xxx'
path_mytest='xxx'

test_cnt=56675
with open(path, "r") as fileDescriptor:
    line = fileDescriptor.readline()
    line = True
    while line:
        line = fileDescriptor.readline()

        if line:
            lineItems = line.strip('\n').split(',')

            imageLabel = lineItems[5:5 + 14]
            if '-1.0' not in imageLabel and test_cnt>0:
                test_cnt-=1
                with open(path_mytest, "a+") as f:
                    f.write(line)
            else:
                with open(path_mytrain, "a+") as f:
                    f.write(line)
