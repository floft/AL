# import python libraries
import datetime
import sys
from datetime import datetime

numsensors = 16


def get_datetime(date, dt_time):
    dt_str = date + ' ' + dt_time
    if '.' not in dt_str:
        dt_str += '.00001'
    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt_obj


if __name__ == "__main__":
    """ Remove duplicate sensor entries. """
    infile = sys.argv[1]
    outfilename = infile + ".unique"
    outfile = open(outfilename, "w")
    count = 0
    dt = None
    buffer = ["" for x in range(numsensors)]
    with open(infile, "r") as file:
        for line in file:
            x = str(str(line).strip()).split(' ', 5)
            prevdt = dt
            dt = get_datetime(x[0], x[1])
            if dt == prevdt and count == 0:
                print('duplicate at', dt)
            else:
                buffer[count] = line
                if count == (numsensors - 1):
                    for i in range(len(buffer)):
                        outfile.write(buffer[i])
                    count = 0
                else:
                    count += 1
    outfile.close()
