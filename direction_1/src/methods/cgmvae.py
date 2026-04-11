from __future__ import annotations

from typing import Any

from ..utils import import_required, to_dense_array


def fit_vae_multilabel(
    method_spec: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
) -> dict[str, Any]:
    """Train a simple unimodal VAE-style multi-label baseline."""
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")
    functional = import_required("torch.nn.functional")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = np.asarray(to_dense_array(X_test), dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    hidden_dims = list(params.get("hidden_dims", [256, 128]))
    latent_dim = int(params.get("latent_dim", 32))
    epochs = int(params.get("epochs", 25))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))
    beta = float(params.get("beta", 0.05))

    class VanillaVAE(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            current = input_dim
            encoder_layers: list[Any] = []
            for width in hidden_dims:
                encoder_layers.append(nn.Linear(current, width))
                encoder_layers.append(nn.ReLU())
                current = width
            self.encoder = nn.Sequential(*encoder_layers)
            self.mu_head = nn.Linear(current, latent_dim)
            self.logvar_head = nn.Linear(current, latent_dim)

            decoder_layers: list[Any] = []
            current = latent_dim
            for width in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(current, width))
                decoder_layers.append(nn.ReLU())
                current = width
            self.decoder = nn.Sequential(*decoder_layers)
            self.classifier = nn.Linear(current, output_dim)

        def reparameterize(self, mu: Any, logvar: Any) -> Any:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            return mu + noise * std

        def forward(self, batch: Any) -> tuple[Any, Any, Any]:
            hidden = self.encoder(batch)
            mu = self.mu_head(hidden)
            logvar = self.logvar_head(hidden)
            latent = self.reparameterize(mu, logvar)
            decoded = self.decoder(latent)
            logits = self.classifier(decoded)
            return latent, logits, _standard_kl(mu, logvar)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaVAE(X_train_np.shape[1], y_train_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = data_utils.TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _, logits, kl = model(features)
            bce = functional.binary_cross_entropy_with_logits(logits, labels)
            loss = bce + beta * kl
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_np, dtype=torch.float32, device=device)
        latent, logits, _ = model(test_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        embeddings = latent.cpu().numpy()

    return {"probabilities": probabilities, "embeddings": embeddings}


def fit_cgmvae(
    method_spec: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
) -> dict[str, Any]:
    """
    Train a claim-oriented C-GMVAE approximation.

    The implementation keeps the key ideas from the paper:
    a learned multimodal prior over labels, a VAE-style latent space, and a contrastive objective
    that pulls feature embeddings toward active labels and away from inactive labels.
    """
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")
    functional = import_required("torch.nn.functional")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = np.asarray(to_dense_array(X_test), dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    hidden_dims = list(params.get("hidden_dims", [256, 128]))
    latent_dim = int(params.get("latent_dim", 32))
    label_embedding_dim = int(params.get("label_embedding_dim", 64))
    epochs = int(params.get("epochs", 25))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))
    beta = float(params.get("beta", 0.05))
    contrastive_weight = float(params.get("contrastive_weight", 0.1))
    temperature = float(params.get("temperature", 0.1))
    use_mixture_prior = bool(params.get("use_mixture_prior", True))
    use_contrastive = bool(params.get("use_contrastive", True))

    class CGMVAE(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            current = input_dim
            encoder_layers: list[Any] = []
            for width in hidden_dims:
                encoder_layers.append(nn.Linear(current, width))
                encoder_layers.append(nn.ReLU())
                current = width
            self.feature_encoder = nn.Sequential(*encoder_layers)
            self.posterior_mu = nn.Linear(current, latent_dim)
            self.posterior_logvar = nn.Linear(current, latent_dim)

            decoder_layers: list[Any] = []
            current = latent_dim
            for width in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(current, width))
                decoder_layers.append(nn.ReLU())
                current = width
            decoder_layers.append(nn.Linear(current, label_embedding_dim))
            self.decoder = nn.Sequential(*decoder_layers)

            self.label_embeddings = nn.Parameter(torch.randn(output_dim, label_embedding_dim))
            self.label_encoder = nn.Sequential(
                nn.Linear(label_embedding_dim, label_embedding_dim),
                nn.ReLU(),
            )
            self.label_mu = nn.Linear(label_embedding_dim, latent_dim)
            self.label_logvar = nn.Linear(label_embedding_dim, latent_dim)

        def reparameterize(self, mu: Any, logvar: Any) -> Any:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            return mu + noise * std

        def encode_labels(self) -> tuple[Any, Any]:
            encoded = self.label_encoder(self.label_embeddings)
            return self.label_mu(encoded), self.label_logvar(encoded)

        def forward(self, features: Any, labels: Any | None = None) -> tuple[Any, Any, Any]:
            hidden = self.feature_encoder(features)
            mu = self.posterior_mu(hidden)
            logvar = self.posterior_logvar(hidden)
            latent = self.reparameterize(mu, logvar)
            feature_embeddings = self.decoder(latent)
            label_embeddings = functional.normalize(self.label_embeddings, dim=1)
            feature_embeddings = functional.normalize(feature_embeddings, dim=1)
            logits = feature_embeddings @ label_embeddings.T
            if labels is None:
                zero = torch.tensor(0.0, device=features.device)
                return feature_embeddings, logits, zero
            if use_mixture_prior:
                kl = _mixture_kl(mu, logvar, latent, *self.encode_labels(), labels)
            else:
                kl = _standard_kl(mu, logvar)
            contrastive = _contrastive_loss(logits, labels, temperature) if use_contrastive else torch.tensor(0.0, device=features.device)
            return feature_embeddings, logits, beta * kl + contrastive_weight * contrastive

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGMVAE(X_train_np.shape[1], y_train_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = data_utils.TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _, logits, regularizer = model(features, labels)
            bce = functional.binary_cross_entropy_with_logits(logits, labels)
            loss = bce + regularizer
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_np, dtype=torch.float32, device=device)
        embeddings, logits, _ = model(test_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        embedding_np = embeddings.cpu().numpy()

    return {"probabilities": probabilities, "embeddings": embedding_np}


def _standard_kl(mu: Any, logvar: Any) -> Any:
    torch = import_required("torch")
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def _contrastive_loss(logits: Any, labels: Any, temperature: float) -> Any:
    torch = import_required("torch")
    valid = labels.sum(dim=1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)

    scaled = logits[valid] / temperature
    positive_mask = labels[valid]
    log_denominator = torch.logsumexp(scaled, dim=1)
    positive_counts = positive_mask.sum(dim=1).clamp(min=1.0)
    positive_term = (scaled * positive_mask).sum(dim=1) / positive_counts
    return -(positive_term - log_denominator).mean()


def _mixture_kl(posterior_mu: Any, posterior_logvar: Any, latent: Any, label_mu: Any, label_logvar: Any, labels: Any) -> Any:
    torch = import_required("torch")
    log_q = _log_normal(latent, posterior_mu, posterior_logvar)

    component_log_probs = []
    for label_idx in range(label_mu.shape[0]):
        component_log_probs.append(_log_normal(latent, label_mu[label_idx], label_logvar[label_idx]))
    stacked = torch.stack(component_log_probs, dim=1)

    positive_mask = labels > 0.5
    safe_counts = positive_mask.sum(dim=1).clamp(min=1.0)
    masked = stacked.masked_fill(~positive_mask, float("-inf"))
    mixture_log_prob = torch.logsumexp(masked, dim=1) - torch.log(safe_counts)

    no_positive = positive_mask.sum(dim=1) == 0
    if no_positive.any():
        standard_log_prob = _log_normal(
            latent[no_positive],
            torch.zeros_like(posterior_mu[no_positive]),
            torch.zeros_like(posterior_logvar[no_positive]),
        )
        mixture_log_prob[no_positive] = standard_log_prob

    return torch.mean(log_q - mixture_log_prob)


def _log_normal(z: Any, mu: Any, logvar: Any) -> Any:
    torch = import_required("torch")
    return -0.5 * (
        ((z - mu) ** 2) / torch.exp(logvar) + logvar + torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
    ).sum(dim=-1)
