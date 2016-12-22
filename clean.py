
def clean_data(fi, fo, header, suffix):
    head = fi.readline().strip("\n").split(",")
    head = [h.strip('"') for h in head]
    for i, h in enumerate(head):
        if h == "nomprov":
            ip = i
    print(ip)
    n = len(head)
    if header:
        fo.write("%s\n" % ",".join(head))

    print(n)
    for line in fi:
        fields = line.strip("\n").split(",")
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov
        assert len(fields) == n
        fields = [field.strip() for field in fields]
        fo.write("%s%s\n" % (",".join(fields), suffix))

with open("../input/all_clean.csv", "w") as f:
    clean_data(open("../input/train_ver2.csv"), f, True, "")
    comma24 = "".join(["," for i in range(24)])
    clean_data(open("../input/test_ver2.csv"), f, False, comma24)
