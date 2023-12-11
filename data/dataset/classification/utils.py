

# region Download and reorganize Imagenet

# ----------------- unzip script -------------

# unzip train data
# tar -xvf ILSVRC2012_img_train.tar
# unzip validation data
# mkdir ILSVRC2012_img_val
# tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val
# unzip development kit
# tar -xzvf ILSVRC2012_devkit_t12.tar.gz

# continue to unzip train data

# dir=./ILSVRC2012_img_train
# for x in `ls $dir/*tar` do
#   filename=`basename $x .tar`
#   mkdir $dir/$filename
#   tar -xvf $x -C $dir/$filename
# done
# rm *.tar

# --------------------------------------------

import scipy
import os
import shutil


def imagenet_val_process(images_dir, devkit_dir):
  """
  move val images to correspongding class folders.
  """
  # load synset, val ground truth and val images list
  synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
  ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
  lines = ground_truth.readlines()
  labels = [int(line[:-1]) for line in lines]

  root, _, filenames = next(os.walk(images_dir))
  for filename in filenames:
      # val image name -> ILSVRC ID -> WIND
      val_id = int(filename.split('.')[0].split('_')[-1])
      ILSVRC_ID = labels[val_id - 1]
      WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
      print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

      # move val images
      output_dir = os.path.join(root, WIND)
      if os.path.isdir(output_dir):
          pass
      else:
          os.mkdir(output_dir)
      shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


# ------ calling code --------
if __name__ == '__main__':
    img_dir = "/data/imagenet/ILSVRC2012_img_val"
    dev_dir = "/data/imagenet/ILSVRC2012_devkit_t12"
    imagenet_val_process(img_dir, dev_dir)

# ----------------------------

# endregion
