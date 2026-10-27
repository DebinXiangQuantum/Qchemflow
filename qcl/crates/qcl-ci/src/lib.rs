use std::collections::{BTreeSet, HashMap};

pub const MAX_DETERMINANTS: usize = 8192;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

struct SparseMatrix {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl SparseMatrix {
    fn from_rows(rows: Vec<HashMap<usize, f64>>) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for row in rows {
            let mut entries: Vec<_> = row.into_iter().filter(|(_, value)| *value != 0.0).collect();
            entries.sort_unstable_by_key(|(column, _)| *column);
            for (column, value) in entries {
                columns.push(column);
                values.push(value);
            }
            offsets.push(columns.len());
        }
        Self {
            offsets,
            columns,
            values,
        }
    }

    fn multiply(&self, vector: &[f64], output: &mut [f64]) {
        for (row, item) in output.iter_mut().enumerate() {
            *item = (self.offsets[row]..self.offsets[row + 1])
                .map(|index| self.values[index] * vector[self.columns[index]])
                .sum();
        }
    }

    fn dense(&self, size: usize) -> Vec<f64> {
        let mut result = vec![0.0; size * size];
        for row in 0..size {
            for index in self.offsets[row]..self.offsets[row + 1] {
                result[row * size + self.columns[index]] = self.values[index];
            }
        }
        result
    }
}

fn combinations(n: usize, k: usize) -> Vec<u64> {
    fn visit(n: usize, left: usize, next: usize, bits: u64, result: &mut Vec<u64>) {
        if left == 0 {
            result.push(bits);
            return;
        }
        for orbital in next..=n - left {
            visit(n, left - 1, orbital + 1, bits | (1 << orbital), result);
        }
    }
    let mut result = Vec::new();
    visit(n, k, 0, 0, &mut result);
    result
}

fn binomial(n: usize, k: usize) -> u128 {
    (1..=k.min(n - k)).fold(1_u128, |value, index| {
        value * (n + 1 - index) as u128 / index as u128
    })
}

fn annihilate(bits: u64, orbital: usize) -> Option<(u64, f64)> {
    let mask = 1_u64 << orbital;
    if bits & mask == 0 {
        return None;
    }
    let sign = if (bits & (mask - 1)).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    Some((bits ^ mask, sign))
}

fn create(bits: u64, orbital: usize) -> Option<(u64, f64)> {
    let mask = 1_u64 << orbital;
    if bits & mask != 0 {
        return None;
    }
    let sign = if (bits & (mask - 1)).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    Some((bits | mask, sign))
}

fn lowest_eigenpair(mut matrix: Vec<f64>, size: usize) -> Result<(f64, Vec<f64>), String> {
    if size == 1 {
        return Ok((matrix[0], vec![1.0]));
    }
    let mut vectors = vec![0.0; size * size];
    for index in 0..size {
        vectors[index * size + index] = 1.0;
    }
    for _ in 0..50 * size * size {
        let mut largest = 0.0_f64;
        let (mut p, mut q) = (0, 1);
        for row in 0..size {
            for column in row + 1..size {
                let value = matrix[row * size + column].abs();
                if value > largest {
                    largest = value;
                    (p, q) = (row, column);
                }
            }
        }
        if largest < 1e-12 {
            let minimum = (0..size)
                .min_by(|&left, &right| {
                    matrix[left * size + left].total_cmp(&matrix[right * size + right])
                })
                .unwrap();
            return Ok((
                matrix[minimum * size + minimum],
                (0..size).map(|row| vectors[row * size + minimum]).collect(),
            ));
        }
        let app = matrix[p * size + p];
        let aqq = matrix[q * size + q];
        let apq = matrix[p * size + q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (s, c) = angle.sin_cos();
        matrix[p * size + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q * size + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p * size + q] = 0.0;
        matrix[q * size + p] = 0.0;
        for row in 0..size {
            let vrp = vectors[row * size + p];
            let vrq = vectors[row * size + q];
            vectors[row * size + p] = c * vrp - s * vrq;
            vectors[row * size + q] = s * vrp + c * vrq;
            if row == p || row == q {
                continue;
            }
            let arp = matrix[row * size + p];
            let arq = matrix[row * size + q];
            matrix[row * size + p] = c * arp - s * arq;
            matrix[p * size + row] = matrix[row * size + p];
            matrix[row * size + q] = s * arp + c * arq;
            matrix[q * size + row] = matrix[row * size + q];
        }
    }
    Err("CI Jacobi eigensolver did not converge".into())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn lowest_lanczos(matrix: &SparseMatrix, size: usize) -> Result<(f64, Vec<f64>), String> {
    let mut first: Vec<f64> = (0..size)
        .map(|index| (((index * 37 + 13) % 101) as f64 - 50.0) / 50.0)
        .collect();
    let norm = dot(&first, &first).sqrt();
    for value in &mut first {
        *value /= norm;
    }
    let mut basis = vec![first];
    let mut diagonal = Vec::new();
    let mut off_diagonal = Vec::new();
    for iteration in 0..size.min(128) {
        let current = &basis[iteration];
        let mut residual = vec![0.0; size];
        matrix.multiply(current, &mut residual);
        if iteration > 0 {
            for (item, previous) in residual.iter_mut().zip(&basis[iteration - 1]) {
                *item -= off_diagonal[iteration - 1] * previous;
            }
        }
        let alpha = dot(current, &residual);
        for (item, value) in residual.iter_mut().zip(current) {
            *item -= alpha * value;
        }
        // Reorthogonalize twice to keep the Ritz residual meaningful near degeneracies.
        for _ in 0..2 {
            for vector in &basis {
                let overlap = dot(vector, &residual);
                for (item, component) in residual.iter_mut().zip(vector) {
                    *item -= overlap * component;
                }
            }
        }
        let beta = dot(&residual, &residual).sqrt();
        diagonal.push(alpha);
        let order = diagonal.len();
        let mut tridiagonal = vec![0.0; order * order];
        for index in 0..order {
            tridiagonal[index * order + index] = diagonal[index];
            if index > 0 {
                tridiagonal[index * order + index - 1] = off_diagonal[index - 1];
                tridiagonal[(index - 1) * order + index] = off_diagonal[index - 1];
            }
        }
        let (energy, vector) = lowest_eigenpair(tridiagonal, order)?;
        if beta * vector[order - 1].abs() < 1e-12 || beta < 1e-12 {
            let mut ground = vec![0.0; size];
            for (weight, direction) in vector.iter().zip(&basis) {
                for (item, component) in ground.iter_mut().zip(direction) {
                    *item += weight * component;
                }
            }
            let norm = dot(&ground, &ground).sqrt();
            for component in &mut ground {
                *component /= norm;
            }
            return Ok((energy, ground));
        }
        off_diagonal.push(beta);
        basis.push(residual.into_iter().map(|value| value / beta).collect());
    }
    Err("CI Lanczos eigensolver did not converge".into())
}

fn build_matrix(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: &[f64],
    eri: &[f64],
    constant: f64,
) -> Result<(SparseMatrix, usize), String> {
    if n_orb == 0 || n_orb > 32 || n_alpha > n_orb || n_beta > n_orb || 2 * n_orb > 64 {
        return Err("Invalid CI orbital count or electron sector".into());
    }
    if h1.len() != n_orb * n_orb
        || eri.len() != n_orb.pow(4)
        || !constant.is_finite()
        || h1.iter().chain(eri).any(|value| !value.is_finite())
    {
        return Err("CI integrals have invalid shape or nonfinite entries".into());
    }
    if binomial(n_orb, n_alpha) * binomial(n_orb, n_beta) > MAX_DETERMINANTS as u128 {
        return Err(format!("CI sector exceeds {MAX_DETERMINANTS} determinants"));
    }
    let states: Vec<u64> = combinations(n_orb, n_alpha)
        .into_iter()
        .flat_map(|alpha| {
            combinations(n_orb, n_beta)
                .into_iter()
                .map(move |beta| alpha | (beta << n_orb))
        })
        .collect();
    let size = states.len();
    let index: HashMap<u64, usize> = states
        .iter()
        .enumerate()
        .map(|(position, bits)| (*bits, position))
        .collect();
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); size];
    for (column, bits) in states.iter().enumerate() {
        *rows[column].entry(column).or_default() += constant;
        for q in 0..2 * n_orb {
            let Some((after_q, sign_q)) = annihilate(*bits, q) else {
                continue;
            };
            let q_spin = q / n_orb;
            let q_spatial = q % n_orb;
            for p_spatial in 0..n_orb {
                let p = q_spin * n_orb + p_spatial;
                if let Some((bra, sign_p)) = create(after_q, p)
                    && let Some(&row) = index.get(&bra)
                {
                    *rows[row].entry(column).or_default() +=
                        sign_q * sign_p * h1[p_spatial * n_orb + q_spatial];
                }
            }
            for s in 0..2 * n_orb {
                let Some((after_s, sign_s)) = annihilate(after_q, s) else {
                    continue;
                };
                let s_spin = s / n_orb;
                let s_spatial = s % n_orb;
                for r_spatial in 0..n_orb {
                    let r = s_spin * n_orb + r_spatial;
                    let Some((after_r, sign_r)) = create(after_s, r) else {
                        continue;
                    };
                    for p_spatial in 0..n_orb {
                        let p = q_spin * n_orb + p_spatial;
                        if let Some((bra, sign_p)) = create(after_r, p)
                            && let Some(&row) = index.get(&bra)
                        {
                            let integral =
                                eri[((p_spatial * n_orb + q_spatial) * n_orb + r_spatial) * n_orb
                                    + s_spatial];
                            *rows[row].entry(column).or_default() +=
                                0.5 * sign_q * sign_s * sign_r * sign_p * integral;
                        }
                    }
                }
            }
        }
    }
    let pairs: BTreeSet<_> = rows
        .iter()
        .enumerate()
        .flat_map(|(row, entries)| {
            entries.keys().filter_map(move |&column| {
                (row != column).then_some((row.min(column), row.max(column)))
            })
        })
        .collect();
    for (row, column) in pairs {
        let left = *rows[row].get(&column).unwrap_or(&0.0);
        let right = *rows[column].get(&row).unwrap_or(&0.0);
        if (left - right).abs() > 1e-8 {
            return Err("CI Hamiltonian is not Hermitian".into());
        }
        let average = 0.5 * (left + right);
        rows[row].insert(column, average);
        rows[column].insert(row, average);
    }
    let matrix = SparseMatrix::from_rows(rows);
    Ok((matrix, size))
}

pub fn ground_state(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: &[f64],
    eri: &[f64],
    constant: f64,
) -> Result<(f64, Vec<f64>), String> {
    let (matrix, size) = build_matrix(n_orb, n_alpha, n_beta, h1, eri, constant)?;
    if size <= 256 {
        lowest_eigenpair(matrix.dense(size), size)
    } else {
        lowest_lanczos(&matrix, size)
    }
}

pub fn sigma(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: &[f64],
    eri: &[f64],
    constant: f64,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    let (matrix, size) = build_matrix(n_orb, n_alpha, n_beta, h1, eri, constant)?;
    if vector.len() != size || vector.iter().any(|value| !value.is_finite()) {
        return Err("CI vector has invalid length or nonfinite entries".into());
    }
    let mut product = vec![0.0; size];
    matrix.multiply(vector, &mut product);
    Ok(product)
}

pub fn ground_energy(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: &[f64],
    eri: &[f64],
    constant: f64,
) -> Result<f64, String> {
    ground_state(n_orb, n_alpha, n_beta, h1, eri, constant).map(|(energy, _)| energy)
}

pub fn rdm1(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    if n_orb == 0 || n_orb > 32 || n_alpha > n_orb || n_beta > n_orb {
        return Err("Invalid CI orbital count or electron sector".into());
    }
    let states: Vec<u64> = combinations(n_orb, n_alpha)
        .into_iter()
        .flat_map(|alpha| {
            combinations(n_orb, n_beta)
                .into_iter()
                .map(move |beta| alpha | (beta << n_orb))
        })
        .collect();
    if vector.len() != states.len() || vector.iter().any(|value| !value.is_finite()) {
        return Err("CI wavefunction has invalid length or nonfinite entries".into());
    }
    let norm = dot(vector, vector);
    if (norm - 1.0).abs() > 1e-8 {
        return Err("CI wavefunction is not normalized".into());
    }
    let index: HashMap<u64, usize> = states
        .iter()
        .enumerate()
        .map(|(position, bits)| (*bits, position))
        .collect();
    let mut density = vec![0.0; n_orb * n_orb];
    for (column, bits) in states.iter().enumerate() {
        for spin in 0..2 {
            for q in 0..n_orb {
                let Some((after_q, sign_q)) = annihilate(*bits, spin * n_orb + q) else {
                    continue;
                };
                for p in 0..n_orb {
                    if let Some((bra, sign_p)) = create(after_q, spin * n_orb + p)
                        && let Some(&row) = index.get(&bra)
                    {
                        density[p * n_orb + q] += sign_q * sign_p * vector[row] * vector[column];
                    }
                }
            }
        }
    }
    Ok(density)
}

pub fn rdm2(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    if n_orb == 0 || n_orb > 32 || n_alpha > n_orb || n_beta > n_orb {
        return Err("Invalid CI orbital count or electron sector".into());
    }
    let states: Vec<u64> = combinations(n_orb, n_alpha)
        .into_iter()
        .flat_map(|alpha| {
            combinations(n_orb, n_beta)
                .into_iter()
                .map(move |beta| alpha | (beta << n_orb))
        })
        .collect();
    if vector.len() != states.len() || vector.iter().any(|value| !value.is_finite()) {
        return Err("CI wavefunction has invalid length or nonfinite entries".into());
    }
    if (dot(vector, vector) - 1.0).abs() > 1e-8 {
        return Err("CI wavefunction is not normalized".into());
    }
    let index: HashMap<u64, usize> = states
        .iter()
        .enumerate()
        .map(|(position, bits)| (*bits, position))
        .collect();
    let mut density = vec![0.0; n_orb.pow(4)];
    // PySCF's chemist order is <a_p^+ a_r^+ a_s a_q> at [p,q,r,s].
    for (column, bits) in states.iter().enumerate() {
        if vector[column] == 0.0 {
            continue;
        }
        for first_spin in 0..2 {
            for q in 0..n_orb {
                let Some((after_q, sign_q)) = annihilate(*bits, first_spin * n_orb + q) else {
                    continue;
                };
                for second_spin in 0..2 {
                    for s in 0..n_orb {
                        let Some((after_s, sign_s)) = annihilate(after_q, second_spin * n_orb + s)
                        else {
                            continue;
                        };
                        for r in 0..n_orb {
                            let Some((after_r, sign_r)) = create(after_s, second_spin * n_orb + r)
                            else {
                                continue;
                            };
                            for p in 0..n_orb {
                                if let Some((bra, sign_p)) = create(after_r, first_spin * n_orb + p)
                                    && let Some(&row) = index.get(&bra)
                                {
                                    density[((p * n_orb + q) * n_orb + r) * n_orb + s] += sign_q
                                        * sign_s
                                        * sign_r
                                        * sign_p
                                        * vector[row]
                                        * vector[column];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(density)
}

#[cfg(test)]
mod tests {
    use super::{ground_energy, ground_state, rdm1, rdm2, sigma};

    #[test]
    fn one_electron_one_orbital() {
        let result = ground_energy(1, 1, 0, &[2.0], &[0.0], 0.5).unwrap();
        assert!((result - 2.5).abs() < 1e-12);
    }

    #[test]
    fn two_orbital_one_electron() {
        let h1 = [1.0, 0.2, 0.2, 2.0];
        let result = ground_energy(2, 1, 0, &h1, &[0.0; 16], 0.0).unwrap();
        let reference = (3.0 - (1.0_f64 + 0.16).sqrt()) / 2.0;
        assert!((result - reference).abs() < 1e-12);
    }

    #[test]
    fn sigma_matches_dense_one_particle_action_and_checks_shape() {
        let h1 = [1.0, 0.2, 0.2, 2.0];
        let result = sigma(2, 1, 0, &h1, &[0.0; 16], 0.5, &[2.0, -1.0]).unwrap();
        assert!((result[0] - 2.8).abs() < 1e-12);
        assert!((result[1] + 2.1).abs() < 1e-12);
        assert!(sigma(2, 1, 0, &h1, &[0.0; 16], 0.5, &[1.0]).is_err());
    }

    #[test]
    fn one_electron_density_tracks_native_eigenvector() {
        let h1 = [1.0, 0.2, 0.2, 2.0];
        let (_, vector) = ground_state(2, 1, 0, &h1, &[0.0; 16], 0.0).unwrap();
        let density = rdm1(2, 1, 0, &vector).unwrap();
        assert!((density[0] + density[3] - 1.0).abs() < 1e-12);
        assert!((density[1] - density[2]).abs() < 1e-12);
        assert!((density[1] - vector[0] * vector[1]).abs() < 1e-12);
    }

    #[test]
    fn two_particle_density_has_expected_pair_count() {
        let density = rdm2(1, 1, 1, &[1.0]).unwrap();
        assert_eq!(density, [2.0]);
    }

    #[test]
    fn large_sector_is_rejected_before_state_generation() {
        let error = ground_energy(12, 6, 6, &[0.0; 144], &[0.0; 20736], 0.0).unwrap_err();
        assert!(error.contains("8192 determinants"));
    }
}
