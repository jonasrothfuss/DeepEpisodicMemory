import numpy as np


def peak_signal_to_noise_ratio(true, pred, color_depth=None):
  """Image quality metric based on maximal signal power vs. power of the noise.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  assert color_depth is not None, "please specify color depth"

  mse = mean_squared_error(true, pred)
  if mse == 0 or mse is None:
    psnr = float('inf')
  else:
    psnr = 10.0 * np.log(np.square(color_depth) / mse)

  return psnr

def mean_squared_error(y, y_pred):
  return np.mean((y.flatten() - y_pred.flatten()) ** 2)


def dnq_metric(sim_matr):
  sim_matr_rad = np.arccos(sim_matr.as_matrix())
  diagonal = np.diagonal(sim_matr_rad)
  diag_mean = np.mean(diagonal)
  np.fill_diagonal(sim_matr_rad, np.zeros(sim_matr_rad.shape[0]))
  non_diag_mean = np.sum(sim_matr_rad) / float(sim_matr_rad.shape[0] * (sim_matr_rad.shape[0] - 1))
  return (1 - diag_mean / non_diag_mean)