use rand::Rng;

#[no_mangle]
extern "C" struct MLP {
    L: usize,
    d: Vec<usize>,
    W: Vec<Vec<Vec<f64>>>,
    X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

#[no_mangle]
extern "C" fn create_mlp(npl: &[usize]) -> MLP {
    let npl_size = npl.len();

    let mut mlp = MLP {
        L: npl_size - 1,
        d: vec![0; npl_size],
        W: vec![vec![vec![0.0; 0]; 0]; 0],
        X: vec![vec![0.0; 0]; 0],
        deltas: vec![vec![0.0; 0]; 0],
    };

    for i in 0..npl_size {
        mlp.d[i] = npl[i];
        mlp.W = vec![vec![vec![0.0; mlp.d[i + 1]]; mlp.d[i]]; mlp.L];
        mlp.X = vec![vec![0.0; mlp.d[i] + 1]; mlp.L + 1];
        mlp.deltas = vec![vec![0.0; mlp.d[i + 1]]; mlp.L];
    }

    for i in 0..mlp.L {
        for j in 0..mlp.d[i + 1] {
            mlp.W[i][j] = vec![0.0; mlp.d[i] + 1];

            for k in 0..mlp.d[i] + 1 {
                mlp.W[i][j][k] = (rand::random::<f64>() / std::f64::MAX) * 2.0 - 1.0;
            }
        }
    }

    for i in 0..mlp.L + 1 {
        mlp.X[i] = vec![0.0; mlp.d[i] + 1];
    }

    mlp
}

#[no_mangle]
extern "C" fn predict(mlp: &mut MLP, inputs: &[f32], is_classification: bool, output: &mut [f32]) {
    if inputs.len() != mlp.d[0] {
        println!("Erreur : la taille des entrées ne correspond pas au nombre de neurones de la première couche.");
        return;
    }

    propagate(mlp, inputs, is_classification);

    let outputs = &mlp.X[mlp.L];
    output.copy_from_slice(&outputs.iter().map(|&x| x as f32).collect::<Vec<f32>>());
}

#[no_mangle]
extern "C" fn train(mlp: &mut MLP, samples_inputs: &[f32], samples_expected_outputs: &[f32],
    samples_size: usize, inputs_size: usize, outputs_size: usize,
    is_classification: bool, iteration_count: usize, alpha: f32) -> f32 {
    let mut loss = 0.0;
    let mut accuracy = 0.0;

    for i in 0..iteration_count {
        let index = rand::thread_rng().gen_range(0..samples_size);

        propagate(mlp, &samples_inputs[index * inputs_size..], is_classification);

        for j in 0..outputs_size {
            mlp.deltas[mlp.L - 1][j] = (samples_expected_outputs[index * outputs_size + j] - mlp.X[mlp.L][j])
                * mlp.X[mlp.L][j] * (1.0 - mlp.X[mlp.L][j]);
            loss += (samples_expected_outputs[index * outputs_size + j] - mlp.X[mlp.L][j]).powi(2);
            if is_classification {
                if (mlp.X[mlp.L][j].round() as i32) == (samples_expected_outputs[index * outputs_size + j] as i32) {
                    accuracy += 1.0;
                }
            }
        }

        for j in (0..mlp.L - 1).rev() {
            for k in 0..mlp.d[j + 1] {
                let mut sum = 0.0;

                for l in 0..mlp.d[j + 2] {
                    sum += mlp.W[j + 1][l][k] * mlp.deltas[j + 1][l];
                }

                mlp.deltas[j][k] = sum * mlp.X[j + 1][k] * (1.0 - mlp.X[j + 1][k]);
            }
        }

        for j in 0..mlp.L {
            for k in 0..mlp.d[j + 1] {
                for l in 0..mlp.d[j] + 1 {
                    if l == mlp.d[j] {
                        mlp.W[j][k][l] += alpha * mlp.deltas[j][k];
                    } else {
                        mlp.W[j][k][l] += alpha * mlp.deltas[j][k] * mlp.X[j][l];
                    }
                }
            }
        }

        if i % 500 == 0 {
            loss = loss / (outputs_size as f32 * 500.0);
            accuracy = accuracy / (outputs_size as f32 * 500.0);
            println!("Iteration {}: Loss = {:.4}, Accuracy = {:.2}%", i, loss, accuracy * 100.0);
            loss = 0.0;
            accuracy = 0.0;
        }
    }

    //save_model(mlp);

    loss
}

#[no_mangle]
extern "C" fn propagate(mlp: &mut MLP, inputs: &[f32], is_classification: bool) {
        for i in 0..mlp.d[0] {
            mlp.X[0][i] = inputs[i];
        }

        mlp.X[0][mlp.d[0]] = 1.0;

        for i in 1..mlp.L + 1 {
            for j in 0..mlp.d[i] {
                let mut sum = 0.0;

                for k in 0..mlp.d[i - 1] + 1 {
                    sum += mlp.W[i - 1][j][k] * mlp.X[i - 1][k];
                }

                mlp.X[i][j] = 1.0 / (1.0 + (-sum).exp());
            }

            if i != mlp.L || !is_classification {
                mlp.X[i][mlp.d[i]] = 1.0;
            }
        }
}
