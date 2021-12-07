import os
import logging
import torch


class MasksSaver:

    def __init__(self, base_directory, name):
        self.directory = os.path.join(base_directory, name)
        self.state = "start"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            logging.debug("directory created: {}".format(self.directory))

        logging.debug("working directory: {}".format(self.directory))
        self.log_journal = os.path.join(self.directory, "journal.txt")
        with open(self.log_journal, "w") as f:
            f.write("start")

    def write_masks(self, index, original, output):
        if self.state == "end":
            logging.error(
                "Write after finish idx: {}, dir: {}".format(index, self.directory))
            raise Exception("Write after finish")

        torch.save(original, os.path.join(self.directory, "original_{}.pth".format(index)))
        torch.save(output, os.path.join(self.directory, "output_{}.pth".format(index)))

    def end(self):
        self.state = "end"
        with open(self.log_journal, "w") as f:
            f.write("end")


if __name__ == "__main__":
    saver = MasksSaver("/Users/nduginets/Desktop/tmp", "train_epoch_1")

    saver.end()
