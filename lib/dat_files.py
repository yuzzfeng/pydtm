import struct


def read(path, fields, record_count):
    fields.sort(key = lambda (name, offset, codec): offset)

    with open(path, "rb") as file_:
        for c in xrange(record_count):
            point = []

            for field in fields:
                name, offset, codec = field
                data = None
                if codec == "F32":
                    binary = file_.read(4)
                    data = struct.unpack("f", binary)[0]
                elif codec == "F64":
                    binary = file_.read(8)
                    data = struct.unpack("d", binary)[0]
                elif codec == "I16":
                    binary = file_.read(2)
                    data = struct.unpack("h", binary)[0]
                elif codec == "I32":
                    binary = file_.read(4)
                    data = struct.unpack("i", binary)[0]
                else:
                    print codec
                point.append(data)

            yield point
