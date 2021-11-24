import json
import logging


class Dumper:

    def __init__(self, path):
        self.path = path
        self.item = {}
        self.__initialize()

    def add_record(self, key, record):
        self.item[key] = record
        with open(self.path, "w") as f:
            f.write(self.get_formatted_data())

    def __initialize(self):
        try:
            with open(self.path, "r") as f:
                data = "\n".join(f.readlines())
                self.item = json.loads(data)
                self.item = {int(k): v for (k, v) in self.item.items()}
        except:
            logging.error("Failed to open file: " + self.path)

    def get_formatted_data(self):
        return json.dumps(self.item, indent=4, sort_keys=True)

    def latest_key(self, or_default=0):
        keys = list(self.item.keys())
        keys.sort()
        return or_default if len(keys) == 0 else keys[-1]


if __name__ == "__main__":
    d = Dumper("ff1.json")
    #d.add_record(1, 1488)
    #d.add_record(0, 23)

    print(d.latest_key())
