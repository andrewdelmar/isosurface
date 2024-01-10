use std::sync::Arc;

use nalgebra::Vector3;

use crate::data::Dimension;

use super::{prod::SDFExprProd, sop::SDFExprSOP};

#[derive(Clone)]
pub(super) enum SDFExprTerm {
    Dim(Dimension),
    GT {
        left: Arc<SDFExprSOP>,
        right: Arc<SDFExprSOP>,
        true_val: Arc<SDFExprSOP>,
        false_val: Arc<SDFExprSOP>,
    },
}

impl SDFExprTerm {
    pub(super) fn eval(&self, at: &Vector3<f64>) -> f64 {
        match self {
            SDFExprTerm::Dim(Dimension::X) => at.x,
            SDFExprTerm::Dim(Dimension::Y) => at.y,
            SDFExprTerm::Dim(Dimension::Z) => at.z,
            SDFExprTerm::GT {
                left,
                right,
                true_val,
                false_val,
            } => {
                if left.eval(at) > right.eval(at) {
                    true_val.eval(at)
                } else {
                    false_val.eval(at)
                }
            }
        }
    }

    pub(super) fn derivative(&self, wrt: &Dimension) -> Option<SDFExprProd> {
        let terms = match self {
            SDFExprTerm::Dim(d) => {
                if d == wrt {
                    Some(Vec::default())
                } else {
                    None
                }
            }
            SDFExprTerm::GT {
                left,
                right,
                true_val,
                false_val,
            } => Some(vec![Self::GT {
                left: left.clone(),
                right: right.clone(),
                true_val: Arc::new(true_val.derivative(wrt)),
                false_val: Arc::new(false_val.derivative(wrt)),
            }]),
        };

        terms.map(|ts| SDFExprProd {
            mul: 1.0,
            terms: ts,
        })
    }
}

impl From<Dimension> for SDFExprTerm {
    fn from(value: Dimension) -> Self {
        Self::Dim(value)
    }
}
