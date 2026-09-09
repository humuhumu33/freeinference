//! The receipt: one answer sealed under its own κ.
//!
//! Shape follows `SessionAttestation` in uor-hologram's realizations: the
//! operands are the κs of the facts the receipt binds, the payload is a
//! detached Ed25519 signature, and the signable bytes are the same encoding
//! with an empty payload. The receipt's own κ is the plain BLAKE3 address of
//! the sealed bytes, exactly what `hologram::space::address_bytes` computes.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Stable identity of this record layout. Changing the fields changes the IRI.
pub const RECEIPT_IRI: &str = "https://freeinference.ai/receipt/v2";
/// Object store kind under which receipts are kept.
pub const RECEIPT_KIND: &str = "receipt";
/// Media type of the stored receipt document.
pub const RECEIPT_MEDIA_TYPE: &str = "application/vnd.freeinference.receipt+json";

/// A κ label as hologram-live and uor-hologram write it: `blake3:` plus 64
/// lowercase hex characters, 71 bytes.
pub fn kappa_of(bytes: &[u8]) -> String {
    let label = hologram::space::address_bytes(bytes);
    String::from_utf8(label.as_bytes().to_vec()).expect("kappa labels are ASCII")
}

/// Facts a receipt binds. Every field is a κ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bound {
    pub model_kappa: String,
    pub engine_kappa: String,
    pub prompt_kappa: String,
    pub params_kappa: String,
    pub output_kappa: String,
    /// κ of the engine's own sealed answer record, when the engine seals
    /// one. Empty when it does not; the position is still bound so the two
    /// cases never share a canonical form.
    #[serde(default)]
    pub answer_kappa: String,
}

impl Bound {
    fn operands(&self) -> [&str; 6] {
        [
            &self.model_kappa,
            &self.engine_kappa,
            &self.prompt_kappa,
            &self.params_kappa,
            &self.output_kappa,
            &self.answer_kappa,
        ]
    }

    /// Canonical encoding: IRI, a zero byte, the operand count, each operand
    /// length prefixed, then the length prefixed payload. Lengths are little
    /// endian u32. Deterministic by construction, no map ordering involved.
    pub fn canonical(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(RECEIPT_IRI.as_bytes());
        out.push(0);
        let operands = self.operands();
        out.extend_from_slice(&(operands.len() as u32).to_le_bytes());
        for operand in operands {
            out.extend_from_slice(&(operand.len() as u32).to_le_bytes());
            out.extend_from_slice(operand.as_bytes());
        }
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// The bytes a signer signs: the canonical form with an empty payload.
    pub fn signable(&self) -> Vec<u8> {
        self.canonical(&[])
    }
}

/// The stored receipt document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub iri: String,
    #[serde(flatten)]
    pub bound: Bound,
    /// Unix milliseconds when the answer was produced. Informational; not
    /// part of the signed bytes, so a receipt's κ does not depend on time.
    pub created_millis: u64,
    /// Ed25519 public key of the signer, lowercase hex.
    pub public_key: String,
    /// Detached Ed25519 signature over [`Bound::signable`], lowercase hex.
    pub signature: String,
    /// κ of the sealed canonical bytes, [`Bound::canonical`] with the
    /// signature as payload.
    pub kappa: String,
}

impl Receipt {
    /// Recomputes the signable bytes and checks the signature and the κ.
    pub fn verify(&self) -> Result<(), String> {
        if self.iri != RECEIPT_IRI {
            return Err(format!("unknown receipt iri {}", self.iri));
        }
        let public_key =
            decode_hex(&self.public_key, 32).map_err(|e| format!("public key: {e}"))?;
        let signature = decode_hex(&self.signature, 64).map_err(|e| format!("signature: {e}"))?;
        let verifying =
            VerifyingKey::from_bytes(&public_key).map_err(|e| format!("public key: {e}"))?;
        let signature = Signature::from_bytes(&signature);
        verifying
            .verify(&self.bound.signable(), &signature)
            .map_err(|_| "signature does not verify over the canonical bytes".to_owned())?;
        let expected = kappa_of(&self.bound.canonical(signature.to_bytes().as_slice()));
        if expected != self.kappa {
            return Err(format!(
                "kappa {} does not match sealed bytes {expected}",
                self.kappa
            ));
        }
        Ok(())
    }
}

/// Holds the daemon's receipt signing key.
pub struct ReceiptSigner {
    key: SigningKey,
}

impl ReceiptSigner {
    /// Loads the key at `path` or creates one from OS randomness. The file
    /// holds the 32 byte secret and nothing else.
    pub fn load_or_create(path: &Path) -> std::io::Result<Self> {
        let secret: [u8; 32] = if path.is_file() {
            let bytes = std::fs::read(path)?;
            bytes.as_slice().try_into().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "signing key must be 32 bytes",
                )
            })?
        } else {
            let mut secret = [0u8; 32];
            getrandom::fill(&mut secret).map_err(std::io::Error::other)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, secret)?;
            secret
        };
        Ok(Self {
            key: SigningKey::from_bytes(&secret),
        })
    }

    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self {
            key: SigningKey::from_bytes(&secret),
        }
    }

    pub fn public_key_hex(&self) -> String {
        encode_hex(self.key.verifying_key().as_bytes())
    }

    /// Seals `bound` into a receipt.
    pub fn seal(&self, bound: Bound, created_millis: u64) -> Receipt {
        let signature = self.key.sign(&bound.signable());
        let sealed = bound.canonical(signature.to_bytes().as_slice());
        Receipt {
            iri: RECEIPT_IRI.to_owned(),
            kappa: kappa_of(&sealed),
            bound,
            created_millis,
            public_key: self.public_key_hex(),
            signature: encode_hex(&signature.to_bytes()),
        }
    }
}

pub fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn decode_hex<const N: usize>(text: &str, expected: usize) -> Result<[u8; N], String> {
    if text.len() != expected * 2 {
        return Err(format!("expected {expected} bytes of hex"));
    }
    let mut out = [0u8; N];
    for (index, chunk) in text.as_bytes().chunks(2).enumerate() {
        let pair = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        out[index] = u8::from_str_radix(pair, 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Bound {
        Bound {
            model_kappa: kappa_of(b"model"),
            engine_kappa: kappa_of(b"engine"),
            prompt_kappa: kappa_of(b"prompt"),
            params_kappa: kappa_of(b"params"),
            output_kappa: kappa_of(b"output"),
            answer_kappa: String::new(),
        }
    }

    #[test]
    fn kappa_matches_plain_blake3() {
        let expected = format!("blake3:{}", blake3::hash(b"hello").to_hex());
        assert_eq!(kappa_of(b"hello"), expected);
        assert_eq!(kappa_of(b"hello").len(), 71);
    }

    #[test]
    fn sealed_receipt_verifies_and_tampering_is_refused() {
        let signer = ReceiptSigner::from_secret([7u8; 32]);
        let receipt = signer.seal(sample(), 1);
        receipt.verify().expect("genuine receipt verifies");

        let mut tampered = receipt.clone();
        tampered.bound.output_kappa = kappa_of(b"other output");
        assert!(tampered.verify().is_err(), "changed output must refuse");

        let mut forged = receipt.clone();
        forged.kappa = kappa_of(b"forged");
        assert!(forged.verify().is_err(), "kappa must match sealed bytes");
    }

    #[test]
    fn kappa_is_independent_of_time() {
        let signer = ReceiptSigner::from_secret([9u8; 32]);
        assert_eq!(
            signer.seal(sample(), 1).kappa,
            signer.seal(sample(), 2).kappa
        );
    }
}
