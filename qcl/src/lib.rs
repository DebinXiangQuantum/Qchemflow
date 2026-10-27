use pyo3::prelude::*;
use qcl_ir::{Device, Graph, Guard, NodeKind, ValueType};
use serde_json::json;

fn solvent_params(
    model: Option<String>,
    name: Option<String>,
    eps: Option<f64>,
) -> PyResult<serde_json::Value> {
    match (model.as_deref(), name, eps) {
        (Some("pcm"), Some(name), Some(eps)) => Ok(json!({"name": name, "eps": eps})),
        (Some("smd"), Some(name), None) => Ok(json!({"model": "smd", "name": name})),
        (None, None, None) => Ok(serde_json::Value::Null),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Solvent model, name and dielectric constant are inconsistent",
        )),
    }
}

#[pyfunction]
fn parse_cluster_json(source: &str) -> PyResult<String> {
    let cluster = qcl_cluster::parse(source).map_err(pyo3::exceptions::PyValueError::new_err)?;
    serde_json::to_string(&cluster)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn ci_ground_energy(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: Vec<f64>,
    eri: Vec<f64>,
    constant: f64,
) -> PyResult<f64> {
    qcl_ci::ground_energy(n_orb, n_alpha, n_beta, &h1, &eri, constant)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn ci_ground_state(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: Vec<f64>,
    eri: Vec<f64>,
    constant: f64,
) -> PyResult<(f64, Vec<f64>)> {
    qcl_ci::ground_state(n_orb, n_alpha, n_beta, &h1, &eri, constant)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn ci_sigma(
    n_orb: usize,
    n_alpha: usize,
    n_beta: usize,
    h1: Vec<f64>,
    eri: Vec<f64>,
    constant: f64,
    vector: Vec<f64>,
) -> PyResult<Vec<f64>> {
    qcl_ci::sigma(n_orb, n_alpha, n_beta, &h1, &eri, constant, &vector)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn ci_rdm1(n_orb: usize, n_alpha: usize, n_beta: usize, vector: Vec<f64>) -> PyResult<Vec<f64>> {
    qcl_ci::rdm1(n_orb, n_alpha, n_beta, &vector).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn ci_rdm2(n_orb: usize, n_alpha: usize, n_beta: usize, vector: Vec<f64>) -> PyResult<Vec<f64>> {
    qcl_ci::rdm2(n_orb, n_alpha, n_beta, &vector).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyclass]
struct GraphBuilder {
    graph: Graph,
}

#[pymethods]
impl GraphBuilder {
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    fn molecule(
        &mut self,
        atoms: String,
        basis: String,
        charge: i32,
        spin: i32,
        unit: String,
        source: String,
    ) -> usize {
        self.graph.add(
            "molecule",
            NodeKind::Host,
            ValueType::Molecule,
            vec![],
            json!({"atoms": atoms, "basis": basis, "charge": charge, "spin": spin, "unit": unit}),
            source,
        )
    }

    fn int2e(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Integrals require a Molecule",
        )?;
        Ok(self.graph.add(
            "int2e",
            NodeKind::Compute,
            ValueType::Eri,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn int1e(&mut self, molecule: usize, kind: String, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Integrals require a Molecule",
        )?;
        Ok(self.graph.add(
            "int1e",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![molecule],
            json!({"kind": kind}),
            source,
        ))
    }

    fn int3c(&mut self, molecule: usize, auxbasis: String, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Three-center integrals require a Molecule",
        )?;
        Ok(self.graph.add(
            "int3c",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![molecule],
            json!({"auxbasis": auxbasis}),
            source,
        ))
    }

    fn cholesky(&mut self, molecule: usize, tolerance: f64, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Cholesky ERI factors require a Molecule",
        )?;
        Ok(self.graph.add(
            "cholesky",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![molecule],
            json!({"tolerance": tolerance}),
            source,
        ))
    }

    fn eigh(&mut self, matrix: usize, metric: Option<usize>, source: String) -> PyResult<usize> {
        self.require_type(matrix, ValueType::Tensor, "eigh requires a Tensor")?;
        let mut inputs = vec![matrix];
        if let Some(metric) = metric {
            self.require_type(metric, ValueType::Tensor, "eigh metric requires a Tensor")?;
            inputs.push(metric);
        }
        Ok(self.graph.add(
            "eigh",
            NodeKind::Compute,
            ValueType::Eigensystem,
            inputs,
            json!({}),
            source,
        ))
    }

    fn orthogonalize(&mut self, metric: usize, source: String) -> PyResult<usize> {
        self.require_type(metric, ValueType::Tensor, "orthogonalize requires a Tensor")?;
        Ok(self.graph.add(
            "orthogonalize",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![metric],
            json!({}),
            source,
        ))
    }

    fn solve(
        &mut self,
        matrix: usize,
        rhs: usize,
        unit: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(matrix, ValueType::Tensor, "solve requires a Tensor matrix")?;
        self.require_type(
            rhs,
            ValueType::Tensor,
            "solve requires a Tensor right-hand side",
        )?;
        Ok(self.graph.add(
            "solve",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![matrix, rhs],
            json!({"unit": unit}),
            source,
        ))
    }

    fn svd(&mut self, matrix: usize, source: String) -> PyResult<usize> {
        self.require_type(matrix, ValueType::Tensor, "svd requires a Tensor")?;
        Ok(self.graph.add(
            "svd",
            NodeKind::Compute,
            ValueType::SvdResult,
            vec![matrix],
            json!({}),
            source,
        ))
    }

    fn svd_projection(
        &mut self,
        result: usize,
        component: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            result,
            ValueType::SvdResult,
            "SVD projection requires an SVD result",
        )?;
        if !matches!(component.as_str(), "u" | "s" | "vh") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown SVD component",
            ));
        }
        Ok(self.graph.add(
            &format!("svd.{component}"),
            NodeKind::Projection,
            ValueType::Tensor,
            vec![result],
            json!({}),
            source,
        ))
    }

    fn transform(
        &mut self,
        integrals: usize,
        coefficients: usize,
        source: String,
    ) -> PyResult<usize> {
        if !matches!(
            self.graph.nodes.get(integrals).map(|node| &node.value_type),
            Some(ValueType::Tensor | ValueType::Eri)
        ) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transform requires integral Tensor or ERI",
            ));
        }
        self.require_type(
            coefficients,
            ValueType::Tensor,
            "transform requires a coefficient Tensor",
        )?;
        Ok(self.graph.add(
            "transform",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![integrals, coefficients],
            json!({}),
            source,
        ))
    }

    fn contract(&mut self, spec: String, tensors: Vec<usize>, source: String) -> PyResult<usize> {
        for tensor in &tensors {
            self.require_type(*tensor, ValueType::Tensor, "Contraction requires Tensors")?;
        }
        Ok(self.graph.add(
            "contract",
            NodeKind::Compute,
            ValueType::Tensor,
            tensors,
            json!({"spec": spec}),
            source,
        ))
    }

    fn periodic_scf(
        &mut self,
        system_json: String,
        kmesh: Vec<u32>,
        method: String,
        exxdiv: Option<String>,
        source: String,
    ) -> PyResult<usize> {
        let system: serde_json::Value = serde_json::from_str(&system_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "periodic.scf",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![],
            json!({"system": system, "kmesh": kmesh, "method": method, "exxdiv": exxdiv}),
            source,
        ))
    }

    fn spectral_exact(&mut self, target_json: String, source: String) -> PyResult<usize> {
        let target: serde_json::Value = serde_json::from_str(&target_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "dynamics.spectral_exact",
            NodeKind::Compute,
            ValueType::SpectralEstimate,
            vec![],
            target,
            source,
        ))
    }

    fn eigh_projection(
        &mut self,
        eigensystem: usize,
        component: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            eigensystem,
            ValueType::Eigensystem,
            "Eigenvalue projection requires an eigensystem",
        )?;
        if component != "values" && component != "vectors" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown eigensystem component",
            ));
        }
        Ok(self.graph.add(
            &format!("eigh.{component}"),
            NodeKind::Projection,
            ValueType::Tensor,
            vec![eigensystem],
            json!({}),
            source,
        ))
    }

    fn gfn2_energy(
        &mut self,
        molecule: usize,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "GFN2 requires a Molecule")?;
        Ok(self.graph.add(
            "gfn2.energy",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![molecule],
            json!({"solvent": solvent_model.zip(solvent_name).map(|(model, name)|
                json!({"model": model, "name": name}))}),
            source,
        ))
    }

    fn gfn2_optimize(
        &mut self,
        molecule: usize,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "GFN2 requires a Molecule")?;
        Ok(self.graph.add(
            "gfn2.optimize",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![molecule],
            json!({"solvent": solvent_model.zip(solvent_name).map(|(model, name)|
                json!({"model": model, "name": name}))}),
            source,
        ))
    }

    fn rhf_with_options(
        &mut self,
        molecule: usize,
        eri: Option<usize>,
        guess: Option<usize>,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        eps: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "RHF requires a Molecule")?;
        let mut inputs = vec![molecule];
        if let Some(eri) = eri {
            self.require_type(eri, ValueType::Eri, "RHF integral input must be ERI")?;
            inputs.push(eri);
        }
        if let Some(guess) = guess {
            self.require_type(
                guess,
                ValueType::ScfResult,
                "RHF guess must be an SCF result",
            )?;
            inputs.push(guess);
        }
        let solvent = solvent_params(solvent_model, solvent_name, eps)?;
        Ok(self.graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            inputs,
            json!({"solvent": solvent, "reuse_eri": eri.is_some(),
                   "reuse_guess": guess.is_some()}),
            source,
        ))
    }

    fn hubbard(
        &mut self,
        geometry: String,
        shape: Vec<usize>,
        hopping: f64,
        repulsion: f64,
        source: String,
    ) -> usize {
        self.graph.add(
            "models.hubbard",
            NodeKind::Host,
            ValueType::FermionOperator,
            vec![],
            json!({"geometry": geometry, "shape": shape, "t": hopping, "U": repulsion}),
            source,
        )
    }

    fn lattice_sector(
        &mut self,
        operator: usize,
        n_alpha: usize,
        n_beta: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "Sector requires an operator",
        )?;
        Ok(self.graph.add(
            "lattice.sector",
            NodeKind::Host,
            ValueType::Sector,
            vec![operator],
            json!({"n_alpha": n_alpha, "n_beta": n_beta}),
            source,
        ))
    }

    fn ed_ground_state(
        &mut self,
        operator: usize,
        sector: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "ED requires an operator",
        )?;
        self.require_type(sector, ValueType::Sector, "ED requires a Sector")?;
        Ok(self.graph.add(
            "ed.ground_state",
            NodeKind::Compute,
            ValueType::GroundState,
            vec![operator, sector],
            json!({}),
            source,
        ))
    }

    fn expect(
        &mut self,
        state: usize,
        observable: String,
        left: usize,
        right: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            state,
            ValueType::GroundState,
            "Expectation requires a ground state",
        )?;
        Ok(self.graph.add(
            "expect",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![state],
            json!({"observable": observable, "left": left, "right": right}),
            source,
        ))
    }

    fn map_group(
        &mut self,
        items: Vec<usize>,
        lanes: Vec<String>,
        collection: Option<usize>,
        source: String,
    ) -> usize {
        let mut inputs = items;
        if let Some(id) = collection {
            inputs.insert(0, id);
        }
        self.graph.add(
            "map",
            NodeKind::Host,
            ValueType::Collection,
            inputs,
            json!({"lanes": lanes, "dynamic": collection.is_some()}),
            source,
        )
    }

    fn scan_group(&mut self, items: Vec<usize>, lanes: Vec<String>, source: String) -> usize {
        self.graph.add(
            "scan",
            NodeKind::Host,
            ValueType::Collection,
            items,
            json!({"lanes": lanes}),
            source,
        )
    }

    fn boolean_constant(&mut self, value: bool, source: String) -> usize {
        self.graph.add(
            "boolean.constant",
            NodeKind::Host,
            ValueType::Boolean,
            vec![],
            json!({"value": value}),
            source,
        )
    }

    fn iterate_group(
        &mut self,
        value: usize,
        done: usize,
        steps: usize,
        lane: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            done,
            ValueType::Boolean,
            "Iteration completion must be boolean",
        )?;
        let value_type = self
            .graph
            .nodes
            .get(value)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown iteration value"))?
            .value_type
            .clone();
        Ok(self.graph.add(
            "iterate",
            NodeKind::Host,
            value_type,
            vec![value, done],
            json!({"steps": steps, "lanes": [lane]}),
            source,
        ))
    }

    fn molecule_edit(
        &mut self,
        molecule: usize,
        kind: String,
        atoms: Vec<usize>,
        value: f64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Geometry edit requires a molecule",
        )?;
        Ok(self.graph.add(
            "molecule.edit",
            NodeKind::Host,
            ValueType::Molecule,
            vec![molecule],
            json!({"kind": kind, "atoms": atoms, "value": value}),
            source,
        ))
    }

    fn constrained_optimize(
        &mut self,
        molecule: usize,
        fixed_json: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Constrained optimization requires a molecule",
        )?;
        let fixed: serde_json::Value = serde_json::from_str(&fixed_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "geometry.constrained_optimize",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![molecule],
            json!({"fixed": fixed}),
            source,
        ))
    }

    fn fd_optimize(
        &mut self,
        molecule: usize,
        fixed_json: String,
        step: f64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Finite-difference optimization requires a molecule",
        )?;
        let fixed: serde_json::Value = serde_json::from_str(&fixed_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "geometry.fd_optimize",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![molecule],
            json!({"fixed": fixed, "step": step}),
            source,
        ))
    }

    fn optimize_group(&mut self, value: usize, lane: String, source: String) -> PyResult<usize> {
        self.require_type(
            value,
            ValueType::Molecule,
            "Optimization result requires a molecule",
        )?;
        Ok(self.graph.add(
            "optimize",
            NodeKind::Host,
            ValueType::Molecule,
            vec![value],
            json!({"lanes": [lane]}),
            source,
        ))
    }

    fn irc(&mut self, transition: usize, source: String) -> PyResult<usize> {
        self.require_type(transition, ValueType::Molecule, "IRC requires a molecule")?;
        Ok(self.graph.add(
            "irc",
            NodeKind::Compute,
            ValueType::ReactionPath,
            vec![transition],
            json!({}),
            source,
        ))
    }

    fn path_connects(
        &mut self,
        path: usize,
        reactant: usize,
        product: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            path,
            ValueType::ReactionPath,
            "Connectivity requires an IRC path",
        )?;
        self.require_type(
            reactant,
            ValueType::Molecule,
            "IRC reactant requires a molecule",
        )?;
        self.require_type(
            product,
            ValueType::Molecule,
            "IRC product requires a molecule",
        )?;
        Ok(self.graph.add(
            "path.connects",
            NodeKind::Host,
            ValueType::Boolean,
            vec![path, reactant, product],
            json!({}),
            source,
        ))
    }

    fn imaginary_count(&mut self, frequencies: usize, source: String) -> PyResult<usize> {
        self.require_type(
            frequencies,
            ValueType::Frequencies,
            "Imaginary count requires frequencies",
        )?;
        Ok(self.graph.add(
            "frequencies.n_imaginary",
            NodeKind::Projection,
            ValueType::Integer,
            vec![frequencies],
            json!({}),
            source,
        ))
    }

    fn integer_compare(
        &mut self,
        value: usize,
        expected: i64,
        equal: bool,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(value, ValueType::Integer, "Comparison requires an integer")?;
        Ok(self.graph.add(
            "integer.compare",
            NodeKind::Host,
            ValueType::Boolean,
            vec![value],
            json!({"expected": expected, "equal": equal}),
            source,
        ))
    }

    fn require_condition(&mut self, condition: usize, source: String) -> PyResult<usize> {
        self.require_type(
            condition,
            ValueType::Boolean,
            "Requirement needs a boolean condition",
        )?;
        Ok(self.graph.add(
            "require",
            NodeKind::Host,
            ValueType::Boolean,
            vec![condition],
            json!({}),
            source,
        ))
    }

    fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    fn guard_nodes(
        &mut self,
        start: usize,
        end: usize,
        condition: usize,
        expected: bool,
    ) -> PyResult<()> {
        self.require_type(
            condition,
            ValueType::Boolean,
            "Conditional guard requires a boolean",
        )?;
        if start <= condition || end > self.graph.nodes.len() || start > end {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid conditional branch range",
            ));
        }
        for node in &mut self.graph.nodes[start..end] {
            node.guards.push(Guard {
                condition,
                expected,
            });
        }
        Ok(())
    }

    fn conditional(
        &mut self,
        condition: usize,
        true_value: usize,
        false_value: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            condition,
            ValueType::Boolean,
            "Conditional selection requires a boolean",
        )?;
        let true_type = self
            .graph
            .nodes
            .get(true_value)
            .map(|node| node.value_type.clone());
        let false_type = self
            .graph
            .nodes
            .get(false_value)
            .map(|node| node.value_type.clone());
        if true_type.is_none() || true_type != false_type {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Conditional branches must return the same value type",
            ));
        }
        Ok(self.graph.add(
            "cond",
            NodeKind::Host,
            true_type.unwrap(),
            vec![condition, true_value, false_value],
            json!({}),
            source,
        ))
    }

    fn rank(&mut self, estimates: Vec<usize>, source: String) -> PyResult<usize> {
        for &estimate in &estimates {
            let node =
                self.graph.nodes.get(estimate).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Unknown ranking value")
                })?;
            if node.value_type != ValueType::Estimate
                && !(estimates.len() == 1 && node.value_type == ValueType::Collection)
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Ranking requires Estimates or one collection",
                ));
            }
        }
        Ok(self.graph.add(
            "rank",
            NodeKind::Host,
            ValueType::Ranking,
            estimates,
            json!({}),
            source,
        ))
    }

    fn mean(&mut self, estimates: Vec<usize>, source: String) -> PyResult<usize> {
        if estimates.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mean requires estimates or one collection",
            ));
        }
        for &estimate in &estimates {
            let node = self
                .graph
                .nodes
                .get(estimate)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown mean input"))?;
            if node.value_type != ValueType::Estimate
                && !(estimates.len() == 1 && node.value_type == ValueType::Collection)
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Mean requires Estimates or one collection",
                ));
            }
        }
        Ok(self.graph.add(
            "estimate.mean",
            NodeKind::Host,
            ValueType::Estimate,
            estimates,
            json!({}),
            source,
        ))
    }

    fn compare(&mut self, left: usize, right: usize, source: String) -> PyResult<usize> {
        self.require_type(left, ValueType::Estimate, "Comparison requires Estimates")?;
        self.require_type(right, ValueType::Estimate, "Comparison requires Estimates")?;
        Ok(self.graph.add(
            "estimate.compare",
            NodeKind::Host,
            ValueType::Comparison,
            vec![left, right],
            json!({"threshold_sigma": 2}),
            source,
        ))
    }

    fn rdkit_embed(
        &mut self,
        smiles: String,
        count: usize,
        seed: u64,
        basis: String,
        source: String,
    ) -> usize {
        self.graph.add(
            "rdkit.embed",
            NodeKind::Compute,
            ValueType::Collection,
            vec![],
            json!({"smiles": smiles, "count": count, "seed": seed, "basis": basis}),
            source,
        )
    }

    fn collection_index(
        &mut self,
        collection: usize,
        index: usize,
        spin: i32,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            collection,
            ValueType::Collection,
            "Index requires a collection",
        )?;
        Ok(self.graph.add(
            "collection.index",
            NodeKind::Projection,
            ValueType::Molecule,
            vec![collection],
            json!({"index": index, "spin": spin}),
            source,
        ))
    }

    fn rdkit_mmff_optimize(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "MMFF requires a conformer")?;
        Ok(self.graph.add(
            "rdkit.mmff_optimize",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn rdkit_dedupe(&mut self, collection: usize, rmsd: f64, source: String) -> PyResult<usize> {
        self.require_type(
            collection,
            ValueType::Collection,
            "Dedupe requires conformers",
        )?;
        Ok(self.graph.add(
            "rdkit.dedupe",
            NodeKind::Compute,
            ValueType::Collection,
            vec![collection],
            json!({"rmsd": rmsd}),
            source,
        ))
    }

    fn solubility(
        &mut self,
        model: String,
        smiles: String,
        melting_point_c: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        if !matches!(model.as_str(), "esol" | "gse") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown solubility model",
            ));
        }
        Ok(self.graph.add(
            &format!("solubility.{model}"),
            NodeKind::Compute,
            ValueType::Estimate,
            vec![],
            json!({"smiles": smiles, "melting_point_C": melting_point_c}),
            source,
        ))
    }

    fn metabolic_sites(
        &mut self,
        molecule: usize,
        model_json: String,
        level: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Metabolic sites require a Molecule",
        )?;
        let model: serde_json::Value = serde_json::from_str(&model_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "metabolism.sites",
            NodeKind::Compute,
            ValueType::SiteScores,
            vec![molecule],
            json!({"model": model, "level": level}),
            source,
        ))
    }

    fn binding_mmgbsa(&mut self, params_json: String, source: String) -> PyResult<usize> {
        let params: serde_json::Value = serde_json::from_str(&params_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "binding.mmgbsa",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![],
            params,
            source,
        ))
    }

    fn geometry_optimize(&mut self, molecule: usize, spin: i32, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Optimization requires a Molecule",
        )?;
        Ok(self.graph.add(
            "geometry.optimize",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![molecule],
            json!({"kind": "minimum", "spin": spin}),
            source,
        ))
    }

    fn sella_saddle(
        &mut self,
        guess: usize,
        reactant: usize,
        product: usize,
        spin: i32,
        source: String,
    ) -> PyResult<usize> {
        for molecule in [guess, reactant, product] {
            self.require_type(molecule, ValueType::Molecule, "Saddle requires Molecules")?;
        }
        Ok(self.graph.add(
            "sella.saddle",
            NodeKind::Compute,
            ValueType::Molecule,
            vec![guess, reactant, product],
            json!({"spin": spin, "verify_irc": true}),
            source,
        ))
    }

    fn frequencies(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Frequencies require a Molecule",
        )?;
        Ok(self.graph.add(
            "frequencies",
            NodeKind::Compute,
            ValueType::Frequencies,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn hessian(
        &mut self,
        molecule: usize,
        xc: Option<String>,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        solvent_eps: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "Hessian requires a molecule")?;
        let solvent = solvent_params(solvent_model, solvent_name, solvent_eps)?;
        Ok(self.graph.add(
            "hessian",
            NodeKind::Compute,
            ValueType::Hessian,
            vec![molecule],
            json!({"xc": xc, "solvent": solvent}),
            source,
        ))
    }

    fn gradient(
        &mut self,
        molecule: usize,
        step: Option<f64>,
        xc: Option<String>,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        solvent_eps: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            molecule,
            ValueType::Molecule,
            "Gradient requires a molecule",
        )?;
        let solvent = solvent_params(solvent_model, solvent_name, solvent_eps)?;
        Ok(self.graph.add(
            "gradient",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![molecule],
            json!({"step": step, "xc": xc, "solvent": solvent}),
            source,
        ))
    }

    fn frequencies_from_hessian(&mut self, matrix: usize, source: String) -> PyResult<usize> {
        self.require_type(matrix, ValueType::Hessian, "Frequencies require a Hessian")?;
        Ok(self.graph.add(
            "frequencies.from_hessian",
            NodeKind::Compute,
            ValueType::Frequencies,
            vec![matrix],
            json!({}),
            source,
        ))
    }

    fn rrho(
        &mut self,
        molecule: usize,
        frequencies: usize,
        quantity: String,
        temperature: f64,
        pressure: f64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "RRHO requires a Molecule")?;
        self.require_type(
            frequencies,
            ValueType::Frequencies,
            "RRHO requires Frequencies",
        )?;
        Ok(self.graph.add(
            "thermo.rrho",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![molecule, frequencies],
            json!({"quantity": quantity, "temperature_K": temperature,
                   "pressure_Pa": pressure}),
            source,
        ))
    }

    fn eyring(&mut self, barrier: usize, temperature: f64, source: String) -> PyResult<usize> {
        self.require_type(barrier, ValueType::Estimate, "Eyring requires an Estimate")?;
        Ok(self.graph.add(
            "thermo.eyring",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![barrier],
            json!({"temperature_K": temperature}),
            source,
        ))
    }

    fn allen_dynes(
        &mut self,
        lambda_ep: f64,
        omega_log_k: f64,
        mu_star: f64,
        source: String,
    ) -> usize {
        self.graph.add(
            "superconductivity.allen_dynes",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![],
            json!({"lambda_ep": lambda_ep, "omega_log_K": omega_log_k, "mu_star": mu_star}),
            source,
        )
    }

    fn rhf(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        let input = self
            .graph
            .nodes
            .get(molecule)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown molecule"))?;
        if input.value_type != ValueType::Molecule {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "RHF requires a Molecule",
            ));
        }
        Ok(self.graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn hf(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        let input = self
            .graph
            .nodes
            .get(molecule)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown molecule"))?;
        if input.value_type != ValueType::Molecule {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "HF requires a Molecule",
            ));
        }
        Ok(self.graph.add(
            "scf.hf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn rohf(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "ROHF requires a Molecule")?;
        Ok(self.graph.add(
            "scf.rohf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn uhf(&mut self, molecule: usize, source: String) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "UHF requires a Molecule")?;
        Ok(self.graph.add(
            "scf.uhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({}),
            source,
        ))
    }

    fn dft_rks(
        &mut self,
        molecule: usize,
        xc: String,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        solvent_eps: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "RKS requires a Molecule")?;
        let solvent = solvent_params(solvent_model, solvent_name, solvent_eps)?;
        Ok(self.graph.add(
            "dft.rks",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({"xc": xc, "solvent": solvent}),
            source,
        ))
    }

    fn dft_uks(
        &mut self,
        molecule: usize,
        xc: String,
        solvent_model: Option<String>,
        solvent_name: Option<String>,
        solvent_eps: Option<f64>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(molecule, ValueType::Molecule, "UKS requires a Molecule")?;
        let solvent = solvent_params(solvent_model, solvent_name, solvent_eps)?;
        Ok(self.graph.add(
            "dft.uks",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            json!({"xc": xc, "solvent": solvent}),
            source,
        ))
    }

    fn mp2(&mut self, mean_field: usize, source: String) -> PyResult<usize> {
        self.require_type(
            mean_field,
            ValueType::ScfResult,
            "MP2 requires an SCF result",
        )?;
        Ok(self.graph.add(
            "mp2",
            NodeKind::Compute,
            ValueType::Mp2Result,
            vec![mean_field],
            json!({}),
            source,
        ))
    }

    fn active_space(
        &mut self,
        result: usize,
        electrons: usize,
        orbitals: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            result,
            ValueType::ScfResult,
            "Active space requires an SCF result",
        )?;
        Ok(self.graph.add(
            "active_space",
            NodeKind::Compute,
            ValueType::ActiveSpace,
            vec![result],
            json!({"electrons": electrons, "orbitals": orbitals}),
            source,
        ))
    }

    fn hamiltonian(&mut self, active: usize, source: String) -> PyResult<usize> {
        self.require_type(
            active,
            ValueType::ActiveSpace,
            "Hamiltonian requires an active space",
        )?;
        Ok(self.graph.add(
            "active.hamiltonian",
            NodeKind::Projection,
            ValueType::FermionOperator,
            vec![active],
            json!({}),
            source,
        ))
    }

    fn sector(&mut self, active: usize, source: String) -> PyResult<usize> {
        self.require_type(
            active,
            ValueType::ActiveSpace,
            "Sector requires an active space",
        )?;
        Ok(self.graph.add(
            "active.sector",
            NodeKind::Projection,
            ValueType::Sector,
            vec![active],
            json!({}),
            source,
        ))
    }

    fn sector_shift(
        &mut self,
        sector: usize,
        n_alpha: usize,
        n_beta: usize,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(sector, ValueType::Sector, "Shift requires a Sector")?;
        Ok(self.graph.add(
            "sector.shift",
            NodeKind::Host,
            ValueType::Sector,
            vec![sector],
            json!({"n_alpha": n_alpha, "n_beta": n_beta}),
            source,
        ))
    }

    fn fci(
        &mut self,
        operator: usize,
        sector: usize,
        engine: String,
        source: String,
    ) -> PyResult<usize> {
        if !matches!(engine.as_str(), "pyscf" | "native") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown FCI engine",
            ));
        }
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "FCI requires a FermionOperator",
        )?;
        self.require_type(sector, ValueType::Sector, "FCI requires a Sector")?;
        Ok(self.graph.add(
            "fci.solve",
            NodeKind::Compute,
            ValueType::FciResult,
            vec![operator, sector],
            if engine == "native" {
                json!({"engine": engine})
            } else {
                json!({})
            },
            source,
        ))
    }

    fn rdm1(&mut self, state: usize, source: String) -> PyResult<usize> {
        self.require_ci_state(state, "rdm1 requires an FCI or SCI result")?;
        Ok(self.graph.add(
            "rdm1",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![state],
            json!({}),
            source,
        ))
    }

    fn ci_coefficients(&mut self, state: usize, source: String) -> PyResult<usize> {
        self.require_type(
            state,
            ValueType::FciResult,
            "Coefficients require an FCI result",
        )?;
        Ok(self.graph.add(
            "fci.coefficients",
            NodeKind::Projection,
            ValueType::Tensor,
            vec![state],
            json!({}),
            source,
        ))
    }

    fn ci_vector(&mut self, sector: usize, values_json: String, source: String) -> PyResult<usize> {
        self.require_type(sector, ValueType::Sector, "CI vector requires a Sector")?;
        let values: serde_json::Value = serde_json::from_str(&values_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "ci.vector",
            NodeKind::Host,
            ValueType::Tensor,
            vec![sector],
            json!({"values": values}),
            source,
        ))
    }

    fn sigma(
        &mut self,
        operator: usize,
        coefficients: usize,
        engine: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "Sigma requires a FermionOperator",
        )?;
        self.require_type(
            coefficients,
            ValueType::Tensor,
            "Sigma requires CI coefficients",
        )?;
        Ok(self.graph.add(
            "sigma",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![operator, coefficients],
            json!({"engine": engine}),
            source,
        ))
    }

    fn rdm2(&mut self, state: usize, source: String) -> PyResult<usize> {
        self.require_ci_state(state, "rdm2 requires an FCI or SCI result")?;
        Ok(self.graph.add(
            "rdm2",
            NodeKind::Compute,
            ValueType::Tensor,
            vec![state],
            json!({}),
            source,
        ))
    }

    fn sci(
        &mut self,
        operator: usize,
        sector: usize,
        select_cutoff: f64,
        total_spin: Option<usize>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "SCI requires a FermionOperator",
        )?;
        self.require_type(sector, ValueType::Sector, "SCI requires a Sector")?;
        Ok(self.graph.add(
            "sci.solve",
            NodeKind::Compute,
            ValueType::SciResult,
            vec![operator, sector],
            json!({"select_cutoff": select_cutoff, "total_spin": total_spin}),
            source,
        ))
    }

    fn hf_kernel(&mut self, active: usize, source: String) -> PyResult<usize> {
        self.require_type(
            active,
            ValueType::ActiveSpace,
            "HF kernel requires an active space",
        )?;
        Ok(self.graph.add(
            "kernel.hf",
            NodeKind::Host,
            ValueType::Kernel,
            vec![active],
            json!({}),
            source,
        ))
    }

    fn ccsd_amplitudes(&mut self, active: usize, source: String) -> PyResult<usize> {
        self.require_type(
            active,
            ValueType::ActiveSpace,
            "CCSD requires an active space",
        )?;
        Ok(self.graph.add(
            "ccsd.amplitudes",
            NodeKind::Compute,
            ValueType::LucjParams,
            vec![active],
            json!({}),
            source,
        ))
    }

    fn lucj_kernel(
        &mut self,
        active: usize,
        params_node: Option<usize>,
        params_json: Option<String>,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            active,
            ValueType::ActiveSpace,
            "LUCJ requires an active space",
        )?;
        let mut inputs = vec![active];
        if let Some(params_node) = params_node {
            self.require_type(
                params_node,
                ValueType::LucjParams,
                "LUCJ requires CCSD parameters",
            )?;
            inputs.push(params_node);
        }
        let params = match params_json {
            Some(source) => serde_json::from_str::<serde_json::Value>(&source)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            None => serde_json::Value::Null,
        };
        Ok(self.graph.add(
            "kernel.lucj",
            NodeKind::Host,
            ValueType::Kernel,
            inputs,
            json!({"params": params}),
            source,
        ))
    }

    fn custom_kernel(
        &mut self,
        n_qubits: usize,
        n_clbits: usize,
        params_json: String,
        gates_json: String,
        source: String,
    ) -> PyResult<usize> {
        let params: serde_json::Value = serde_json::from_str(&params_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let gates: serde_json::Value = serde_json::from_str(&gates_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(self.graph.add(
            "kernel.custom",
            NodeKind::Host,
            ValueType::Kernel,
            vec![],
            json!({"n_qubits": n_qubits, "n_clbits": n_clbits,
                   "params": params, "gates": gates}),
            source,
        ))
    }

    fn sample(
        &mut self,
        kernel: usize,
        shots: usize,
        seed: u64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(kernel, ValueType::Kernel, "Sampling requires a Kernel")?;
        Ok(self.graph.add(
            "sample",
            NodeKind::Compute,
            ValueType::Bitstrings,
            vec![kernel],
            json!({"shots": shots, "seed": seed}),
            source,
        ))
    }

    fn estimate_quantum(
        &mut self,
        operator: usize,
        kernel: usize,
        n_groups: usize,
        shots_per_group: usize,
        seed: u64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::QubitOperator,
            "Pauli estimate requires a QubitOperator",
        )?;
        self.require_type(
            kernel,
            ValueType::Kernel,
            "Pauli estimate requires a Kernel",
        )?;
        Ok(self.graph.add(
            "estimate.quantum",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![operator, kernel],
            json!({"n_groups": n_groups, "shots_per_group": shots_per_group,
                   "seed": seed}),
            source,
        ))
    }

    fn vqe_sampled(
        &mut self,
        operator: usize,
        sector: usize,
        ansatz: String,
        optimizer: String,
        max_evaluations: usize,
        shots_per_group: usize,
        n_groups: usize,
        seed: u64,
        source: String,
    ) -> PyResult<usize> {
        let operator_type = self.graph.nodes.get(operator).map(|node| &node.value_type);
        if !matches!(
            operator_type,
            Some(ValueType::FermionOperator | ValueType::QubitOperator)
        ) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "VQE requires a FermionOperator or QubitOperator",
            ));
        }
        self.require_type(sector, ValueType::Sector, "VQE requires a Sector")?;
        Ok(self.graph.add(
            "vqe.sampled",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![operator, sector],
            json!({"ansatz": ansatz, "optimizer": optimizer,
                   "max_evaluations": max_evaluations,
                   "shots_per_group": shots_per_group,
                   "n_groups": n_groups, "seed": seed}),
            source,
        ))
    }

    fn qubit_hamiltonian(
        &mut self,
        operator: usize,
        mapping: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "Jordan-Wigner mapping requires a FermionOperator",
        )?;
        Ok(self.graph.add(
            "qubit_hamiltonian",
            NodeKind::Compute,
            ValueType::QubitOperator,
            vec![operator],
            json!({"mapping": mapping}),
            source,
        ))
    }

    fn split_sample(
        &mut self,
        value: usize,
        devices_json: String,
        shots: Vec<usize>,
        source: String,
    ) -> PyResult<usize> {
        let devices: Vec<Device> = serde_json::from_str(&devices_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let original = self
            .graph
            .nodes
            .get(value)
            .filter(|node| node.op == "sample" && node.device.is_none())
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Expected unplaced sample"))?;
        let total = original.params["shots"].as_u64().unwrap_or(0);
        let seed = original.params["seed"].as_u64().unwrap_or(0);
        if devices.len() < 2
            || devices.len() != shots.len()
            || shots.iter().any(|count| *count == 0)
            || shots.iter().map(|count| *count as u128).sum::<u128>() != u128::from(total)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid sample device or shot split",
            ));
        }
        let mut inputs = vec![value];
        for (index, (device, count)) in devices.into_iter().zip(shots).enumerate() {
            let node_seed = seed
                .checked_add(index as u64)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Seed overflow"))?;
            let params = json!({"shots": count, "seed": node_seed, "repetition": index});
            if index == 0 {
                self.graph.nodes[value].params = params;
                self.graph.nodes[value].device = Some(device);
            } else {
                let id = self.graph.add(
                    "sample",
                    NodeKind::Compute,
                    ValueType::Bitstrings,
                    original.inputs.clone(),
                    params,
                    original.source.clone(),
                );
                self.graph.nodes[id].device = Some(device);
                inputs.push(id);
            }
        }
        Ok(self.graph.add(
            "samples.merge",
            NodeKind::Host,
            ValueType::Bitstrings,
            inputs,
            json!({"shots": total}),
            source,
        ))
    }

    fn split_quantum_estimate(
        &mut self,
        value: usize,
        devices_json: String,
        source: String,
    ) -> PyResult<usize> {
        let devices: Vec<Device> = serde_json::from_str(&devices_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let original = self
            .graph
            .nodes
            .get(value)
            .filter(|node| node.op == "estimate.quantum" && node.device.is_none())
            .cloned()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Expected unplaced Pauli estimate")
            })?;
        let groups = original.params["n_groups"].as_u64().unwrap_or(0) as usize;
        if devices.len() < 2 || devices.len() > groups {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid Pauli measurement-group split",
            ));
        }
        let mut inputs = vec![value];
        let mut offset = 0;
        let lanes = devices.len();
        for (index, device) in devices.into_iter().enumerate() {
            let count = groups / lanes + usize::from(index < groups % lanes);
            let indices: Vec<usize> = (offset..offset + count).collect();
            offset += count;
            let mut params = original.params.clone();
            params["group_indices"] = json!(indices);
            if index == 0 {
                self.graph.nodes[value].params = params;
                self.graph.nodes[value].device = Some(device);
            } else {
                let id = self.graph.add(
                    "estimate.quantum",
                    NodeKind::Compute,
                    ValueType::Estimate,
                    original.inputs.clone(),
                    params,
                    original.source.clone(),
                );
                self.graph.nodes[id].device = Some(device);
                inputs.push(id);
            }
        }
        Ok(self.graph.add(
            "estimates.merge",
            NodeKind::Host,
            ValueType::Estimate,
            inputs,
            json!({"n_groups": groups}),
            source,
        ))
    }

    fn sqd(
        &mut self,
        operator: usize,
        bitstrings: usize,
        sector: usize,
        seed: u64,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            operator,
            ValueType::FermionOperator,
            "SQD requires a FermionOperator",
        )?;
        self.require_type(bitstrings, ValueType::Bitstrings, "SQD requires Bitstrings")?;
        self.require_type(sector, ValueType::Sector, "SQD requires a Sector")?;
        Ok(self.graph.add(
            "sqd.diagonalize",
            NodeKind::Compute,
            ValueType::Estimate,
            vec![operator, bitstrings, sector],
            json!({"seed": seed}),
            source,
        ))
    }

    fn energy(&mut self, result: usize, source: String) -> PyResult<usize> {
        let input = self
            .graph
            .nodes
            .get(result)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown result"))?;
        if !matches!(
            input.value_type,
            ValueType::ScfResult
                | ValueType::Mp2Result
                | ValueType::FciResult
                | ValueType::SciResult
                | ValueType::GroundState
        ) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Energy requires an SCF, MP2, CI or ground-state result",
            ));
        }
        let op = match input.value_type {
            ValueType::FciResult => "fci.energy",
            ValueType::SciResult => "sci.energy",
            ValueType::GroundState => "ed.energy",
            ValueType::Mp2Result => "mp2.energy",
            _ => "scf.energy",
        };
        Ok(self.graph.add(
            op,
            NodeKind::Projection,
            ValueType::Estimate,
            vec![result],
            json!({}),
            source,
        ))
    }

    fn orbital_energy(
        &mut self,
        result: usize,
        orbital: String,
        source: String,
    ) -> PyResult<usize> {
        self.require_type(
            result,
            ValueType::ScfResult,
            "Orbitals require an SCF result",
        )?;
        if !matches!(
            orbital.as_str(),
            "homo" | "lumo" | "homo_alpha" | "homo_beta" | "lumo_alpha" | "lumo_beta"
        ) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown orbital projection",
            ));
        }
        Ok(self.graph.add(
            &format!("scf.{orbital}"),
            NodeKind::Projection,
            ValueType::Estimate,
            vec![result],
            json!({}),
            source,
        ))
    }

    fn estimate_binary(
        &mut self,
        operation: String,
        left: usize,
        right: usize,
        source: String,
    ) -> PyResult<usize> {
        if !matches!(operation.as_str(), "add" | "sub") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown estimate operation",
            ));
        }
        self.require_type(
            left,
            ValueType::Estimate,
            "Left operand must be an Estimate",
        )?;
        self.require_type(
            right,
            ValueType::Estimate,
            "Right operand must be an Estimate",
        )?;
        Ok(self.graph.add(
            &format!("estimate.{operation}"),
            NodeKind::Host,
            ValueType::Estimate,
            vec![left, right],
            json!({}),
            source,
        ))
    }

    fn convert_unit(&mut self, value: usize, unit: String, source: String) -> PyResult<usize> {
        self.require_type(
            value,
            ValueType::Estimate,
            "Unit conversion requires an Estimate",
        )?;
        Ok(self.graph.add(
            "estimate.in_units",
            NodeKind::Host,
            ValueType::Estimate,
            vec![value],
            json!({"unit": unit}),
            source,
        ))
    }

    fn estimate_scale(&mut self, value: usize, factor: f64, source: String) -> PyResult<usize> {
        self.require_type(value, ValueType::Estimate, "Scaling requires an Estimate")?;
        if !factor.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Estimate scale must be finite",
            ));
        }
        Ok(self.graph.add(
            "estimate.scale",
            NodeKind::Host,
            ValueType::Estimate,
            vec![value],
            json!({"factor": factor}),
            source,
        ))
    }

    fn place(&mut self, value: usize, device_json: String, source: String) -> PyResult<usize> {
        let device: Device = serde_json::from_str(&device_json)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        self.graph
            .place(value, device, source)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn diagnostics_json(&self) -> PyResult<String> {
        serde_json::to_string(&qcl_check::check(&self.graph))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    fn graph_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.graph)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }
}

impl GraphBuilder {
    fn require_ci_state(&self, id: usize, message: &str) -> PyResult<()> {
        let node = self
            .graph
            .nodes
            .get(id)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown value"))?;
        if !matches!(node.value_type, ValueType::FciResult | ValueType::SciResult) {
            return Err(pyo3::exceptions::PyTypeError::new_err(message.to_owned()));
        }
        Ok(())
    }

    fn require_type(&self, id: usize, expected: ValueType, message: &str) -> PyResult<()> {
        let node = self
            .graph
            .nodes
            .get(id)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Unknown value"))?;
        if node.value_type != expected {
            return Err(pyo3::exceptions::PyTypeError::new_err(message.to_owned()));
        }
        Ok(())
    }
}

#[pymodule]
fn _qcl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GraphBuilder>()?;
    module.add_function(wrap_pyfunction!(parse_cluster_json, module)?)?;
    module.add_function(wrap_pyfunction!(ci_ground_energy, module)?)?;
    module.add_function(wrap_pyfunction!(ci_ground_state, module)?)?;
    module.add_function(wrap_pyfunction!(ci_sigma, module)?)?;
    module.add_function(wrap_pyfunction!(ci_rdm1, module)?)?;
    module.add_function(wrap_pyfunction!(ci_rdm2, module)?)?;
    module.add("CI_MAX_DETERMINANTS", qcl_ci::MAX_DETERMINANTS)?;
    module.add("CI_VERSION", qcl_ci::VERSION)?;
    Ok(())
}
