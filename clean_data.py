import re
with open("poker-hand.data", "r") as file:
    data = file.readlines()

    occurances = [0,0,0,0,0,0,0,0,0,0]
    output = ""

    for line in data:
        i = int(re.search('^([\d]+,)*(\d)', line).group(2))
        occurances[i] += 1

        if occurances[0] % 1 == 0 and i == 0:
            output += line
        elif occurances[1] % 1 == 0 and i == 1:
            output += line
        elif occurances[2] > 0 and i == 2:
            output += line
        elif occurances[3] > 0 and i == 3:
            output += line
            output += line
        elif occurances[4] > 0 and i == 4:
            for x in range(13):
                output += line
        elif occurances[5] > 0 and i == 5:
            for x in range(25):
                output += line
        elif occurances[6] > 0 and i == 6:
            for x in range(34):
                output += line
        elif occurances[7] > 0 and i == 7:
            for x in range(215):
                output += line
        elif occurances[8] > 0 and i == 8:
            for x in range(2942):
                output += line
        elif occurances[9] > 0 and i == 9:
            for x in range(6250):
                output += line

    print(occurances)

    output_lines = output.split()
    occurances2 = [0,0,0,0,0,0,0,0,0,0]
    for oLine in output_lines:
        i = int(re.search('^([\d]+,)*(\d)', oLine).group(2))
        occurances2[i] += 1
    print(occurances2)

    f = open("poker-hand-test.data", "w")
    f.write(output)
    
    f.close()
    file.close()