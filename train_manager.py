import train_model_multi_gpu


def repeat_training():
  """"""
  for i in range(5000):
    train_model_multi_gpu.main()


repeat_training()

