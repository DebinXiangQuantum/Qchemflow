use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Cpu {
    pub id: String,
    pub threads: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub id: String,
    pub memory_gb: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Simulator {
    pub method: String,
    pub host: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Qpu {
    pub id: String,
    pub qubits: Option<u32>,
    pub site: Option<String>,
    pub queue: Option<String>,
    pub billing: Option<String>,
    pub billing_account: Option<String>,
    pub provider: Option<String>,
    pub backend: Option<String>,
    #[serde(default)]
    pub native_gates: Vec<String>,
    pub calibration: Option<String>,
    pub simulator: Option<Simulator>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub between: [String; 2],
    pub kind: String,
    #[serde(rename = "bandwidth_GBps")]
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantumLink {
    pub between: [String; 2],
    pub bell_pairs_per_s: f64,
    pub fidelity: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Cluster {
    #[serde(default)]
    pub cpu: Vec<Cpu>,
    #[serde(default)]
    pub gpu: Vec<Gpu>,
    #[serde(default)]
    pub qpu: Vec<Qpu>,
    #[serde(default)]
    pub link: Vec<Link>,
    #[serde(default)]
    pub quantum_link: Vec<QuantumLink>,
}

pub fn parse(source: &str) -> Result<Cluster, String> {
    let cluster: Cluster = toml::from_str(source).map_err(|error| error.to_string())?;
    cluster.validate()?;
    Ok(cluster)
}

impl Cluster {
    pub fn validate(&self) -> Result<(), String> {
        if self.cpu.is_empty() {
            return Err("A cluster requires a host CPU".into());
        }
        let mut ids = HashSet::new();
        for id in self
            .cpu
            .iter()
            .map(|device| &device.id)
            .chain(self.gpu.iter().map(|device| &device.id))
            .chain(self.qpu.iter().map(|device| &device.id))
        {
            if id.is_empty() || !ids.insert(id.as_str()) {
                return Err(format!("Duplicate or empty device id: {id}"));
            }
        }
        for cpu in &self.cpu {
            if cpu.threads == Some(0) {
                return Err(format!("{} has no CPU threads", cpu.id));
            }
        }
        for gpu in &self.gpu {
            if !gpu.memory_gb.is_finite() || gpu.memory_gb <= 0.0 {
                return Err(format!("{} has invalid GPU memory", gpu.id));
            }
        }
        for qpu in &self.qpu {
            if qpu.simulator.is_none() && qpu.qubits.unwrap_or(0) == 0 {
                return Err(format!("{} requires a positive qubit count", qpu.id));
            }
            if let Some(simulator) = &qpu.simulator
                && !self.cpu.iter().any(|device| device.id == simulator.host)
                && !self.gpu.iter().any(|device| device.id == simulator.host)
            {
                return Err(format!(
                    "{} has unknown simulator host {}",
                    qpu.id, simulator.host
                ));
            }
            match qpu.billing.as_deref().unwrap_or("none") {
                "none" => {}
                "metered" if qpu.site.as_deref() == Some("remote") && qpu.simulator.is_none() => {}
                _ => return Err(format!("{} has invalid QPU billing mode", qpu.id)),
            }
        }
        for link in &self.link {
            if link.between[0] == link.between[1]
                || link.between.iter().any(|id| !ids.contains(id.as_str()))
            {
                return Err(format!(
                    "Invalid link endpoint: {} -> {}",
                    link.between[0], link.between[1]
                ));
            }
            if !link.bandwidth_gbps.is_finite()
                || link.bandwidth_gbps <= 0.0
                || !link.latency_us.is_finite()
                || link.latency_us < 0.0
            {
                return Err("Link bandwidth and latency must be finite and nonnegative".into());
            }
        }
        let qpu_ids: HashSet<_> = self.qpu.iter().map(|device| device.id.as_str()).collect();
        for link in &self.quantum_link {
            if link.between[0] == link.between[1]
                || link.between.iter().any(|id| !qpu_ids.contains(id.as_str()))
                || !link.bell_pairs_per_s.is_finite()
                || link.bell_pairs_per_s <= 0.0
                || !link.fidelity.is_finite()
                || !(0.0..=1.0).contains(&link.fidelity)
            {
                return Err(format!(
                    "Invalid quantum link: {} -> {}",
                    link.between[0], link.between[1]
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn remote_is_not_automatically_billed() {
        let cluster =
            parse("[[cpu]]\nid='cpu0'\n[[qpu]]\nid='q0'\nqubits=8\nsite='remote'\n").unwrap();
        assert_eq!(cluster.qpu[0].billing, None);
    }

    #[test]
    fn metered_local_qpu_is_rejected() {
        let error = parse(
            "[[cpu]]\nid='cpu0'\n[[qpu]]\nid='q0'\nqubits=8\nsite='local'\nbilling='metered'\n",
        )
        .unwrap_err();
        assert!(error.contains("billing"));
    }
}
