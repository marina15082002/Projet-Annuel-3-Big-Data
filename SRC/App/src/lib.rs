use rand::{thread_rng, Rng}; // Bibliothèque de génération de nombres aléatoires
use rand_distr::Uniform; // Génération de nombres aléatoires selon une distribution uniforme

#[no_mangle]
extern "C" fn predict_labels(points: *const f32, labels: *const f32, points_len: i32, labels_len: i32) -> *mut *mut f32 {

       let points_slice = unsafe { std::slice::from_raw_parts(points, (points_len * 2) as usize) };
       let labels_slice = unsafe { std::slice::from_raw_parts(labels, labels_len as usize) };

       let mut rng = rand::thread_rng();
       let mut w:[f32;3] = [rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];

       for _ in 0..10000 {
           let k = rng.gen_range(0..labels_len as usize);
           let yk = labels_slice[k];
           let xk= [1., points_slice[k * 2], points_slice[k * 2 + 1]];
           let signal = w[0] + w[1] * xk[1] + w[2] * xk[2];
           let gxk = if signal >= 0. { 1. } else { -1. };
           w[0] += 0.01 * (yk - gxk) * xk[0];
           w[1] += 0.01 * (yk - gxk) * xk[1];
           w[2] += 0.01 * (yk - gxk) * xk[2];
       }

       let mut predicted_labels = Vec::new();
       let mut predicted_x1 = Vec::new();
       let mut predicted_x2 = Vec::new();
       for x1 in 0..300 {
           for x2 in 0..300 {
               predicted_x1.push(x1 as f32 / 100.);
               predicted_x2.push(x2 as f32 / 100.);
               let label = if x1 as f32 / 100. * w[1] + x2 as f32 / 100. * w[2] + w[0] >= 0. {
                   0.0
               } else {
                   1.0
               };
                predicted_labels.push(label);
           }
       }

       let predicted_labels_ptr = predicted_labels.as_mut_ptr();
       let predicted_x1_ptr = predicted_x1.as_mut_ptr();
       let predicted_x2_ptr = predicted_x2.as_mut_ptr();
        //println!("predicted_labels: {:?}", predicted_labels);
        println!("predicted_labels.as_mut_ptr: {:?}",  predicted_labels.as_mut_ptr());

        let mut tab2 = Vec::new();
        tab2.push(predicted_x1_ptr);
        tab2.push(predicted_x2_ptr);
        tab2.push(predicted_labels_ptr);

        // predicted_x1_ptr , predicted_x2_ptr et predicted_labels_ptr sont des pointeurs vers les tableaux
        println!("predicted_x1_ptr: {:?}", predicted_x1_ptr);
        println!("predicted_x2_ptr: {:?}", predicted_x2_ptr);
        println!("predicted_labels_ptr: {:?}", predicted_labels_ptr);
        // afficher toutes les valeurs du tableau predicted_labels depuis le tableau tab


        //let tab_ptr = Box::into_raw(Box::new(tab));
        //tab_ptr
        //tab2.as_mut_ptr()
    0 as *mut *mut f32
   }