class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
            except ValueError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass