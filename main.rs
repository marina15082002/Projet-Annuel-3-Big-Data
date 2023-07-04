use rand::Rng;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct myMLP {
    d: Vec<usize>,
    L: usize,
    W: Vec<Array2<f64>>,
    X: Vec<Array2<f64, ndarray::Dim<[usize; 1]>>>,
    deltas: Vec<Array2<f64, ndarray::Dim<[usize; 1]>>>,
}

impl myMLP {
    pub fn new(npl: &[usize]) -> Self {
        let d = npl.to_vec();
        let L = npl.len() - 1;
        let mut W = Vec::new();

        for l in 0..=L {
            W.push(Array::from_elem((npl[l - 1] + 1, npl[l] + 1), 0.0));
            if l == 0 {
                continue;
            }
            for i in 0..npl[l - 1] {
                for j in 0..=npl[l] {
                    W[l][(i,j)] = if j == 0 {
                        0.0
                    } else {
                        rand::thread_rng().sample(Uniform::new(-1.0,1.0))
                    };
                }
            }
        }
    }
    let mut X = Vec::new();
    for l in 0..=L {
        X.push(Array::from_elem(npl[l] + 1, 0.0));
            for j in 0..=npl[l] {
                X[l][j] = if j == 0 { 1.0 } else { 0.0 };
            }
    }
    let mut deltas = Vec::new();
    for l in 0..=L {
        deltas.push(Array::from_elem(npl[l] + 1, 0.0));
    }

    MyMLP { d, L, W, X, deltas }
}
fn propagate(&mut self, inputs: &[f64], is_classification: bool) {
        for (j, &input) in inputs.iter().enumerate() {
            self.X[0][j + 1] = input;
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l] {
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][(i, j)] * self.X[l - 1][i];
                }

                if l < self.L || is_classification {
                    total = total.tanh();
                }

                self.X[l][j] = total;
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.X[self.L][1..].to_vec()
    }

    pub fn train(
        &mut self,
        all_samples_inputs: &[Vec<f64>],
        all_samples_expected_outputs: &[Vec<f64>],
        is_classification: bool,
        iteration_count: usize,
        alpha: f64,
    ) {
        for _ in 0..iteration_count {
            let k = rand::thread_rng().gen_range(0, all_samples_inputs.len());
            let inputs_k = &all_samples_inputs[k];
            let y_k = &all_samples_expected_outputs[k];

            self.propagate(inputs_k, is_classification);

            for j in 1..=self.d[self.L] {
                self.deltas[self.L][j] = self.X[self.L][j] - y_k[j - 1];
                if is_classification {
                    self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                }
            }

            for l in (1..=self.L).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.W[l][(i, j)] * self.deltas[l][j];
                    }
                    self.deltas[l - 1][i] = (1.0 - self.X[l - 1][i].powi(2)) * total;
                }
            }

            for l in 1..=self.L {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.W[l][(i, j)] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
    }

