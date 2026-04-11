from __future__ import annotations

from typing import Any

from ..utils import import_required, to_dense_array


def fit_autoencoder_reducer(method_spec: dict[str, Any], X_train: Any, X_test: Any | None = None) -> tuple[Any, Any | None]:
    """Train an autoencoder and return encoder outputs for train and test."""
    architecture = method_spec.get("params", {}).get("architecture", "mlp")
    if architecture == "cnn":
        return _fit_cnn_autoencoder_reducer(method_spec, X_train, X_test)

    return _fit_mlp_autoencoder_reducer(method_spec, X_train, X_test)


def _fit_mlp_autoencoder_reducer(
    method_spec: dict[str, Any],
    X_train: Any,
    X_test: Any | None = None,
) -> tuple[Any, Any | None]:
    """Train a simple MLP autoencoder and return encoder outputs for train and test."""
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = None if X_test is None else np.asarray(to_dense_array(X_test), dtype=np.float32)

    hidden_dims = list(params.get("hidden_dims", [256, 128]))
    latent_dim = int(params.get("latent_dim", 2))
    epochs = int(params.get("epochs", 20))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))

    mean = X_train_np.mean(axis=0, keepdims=True)
    std = X_train_np.std(axis=0, keepdims=True) + 1e-6
    X_train_scaled = (X_train_np - mean) / std
    X_test_scaled = None if X_test_np is None else (X_test_np - mean) / std

    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()

            encoder_layers: list[Any] = []
            current = input_dim
            for width in hidden_dims:
                encoder_layers.append(nn.Linear(current, width))
                encoder_layers.append(nn.ReLU())
                current = width
            encoder_layers.append(nn.Linear(current, latent_dim))

            decoder_layers: list[Any] = []
            current = latent_dim
            for width in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(current, width))
                decoder_layers.append(nn.ReLU())
                current = width
            decoder_layers.append(nn.Linear(current, input_dim))

            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, batch: Any) -> tuple[Any, Any]:
            latent = self.encoder(batch)
            reconstruction = self.decoder(latent)
            return latent, reconstruction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(X_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = data_utils.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

    def encode(array: Any) -> Any:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(array, dtype=torch.float32, device=device)
            latent, _ = model(tensor)
            return latent.cpu().numpy()

    train_embedding = encode(X_train_scaled)
    test_embedding = None if X_test_scaled is None else encode(X_test_scaled)
    return train_embedding, test_embedding


def _fit_cnn_autoencoder_reducer(
    method_spec: dict[str, Any],
    X_train: Any,
    X_test: Any | None = None,
) -> tuple[Any, Any | None]:
    """Train a lightweight CNN autoencoder for flattened image inputs."""
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = None if X_test is None else np.asarray(to_dense_array(X_test), dtype=np.float32)

    channels, height, width = _infer_image_shape(X_train_np.shape[1], params)
    conv_channels = list(params.get("conv_channels", [16, 32]))
    latent_dim = int(params.get("latent_dim", 2))
    epochs = int(params.get("epochs", 20))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))

    X_train_img = _reshape_flat_images(X_train_np, channels, height, width)
    X_test_img = None if X_test_np is None else _reshape_flat_images(X_test_np, channels, height, width)

    mean = X_train_img.mean(axis=0, keepdims=True)
    std = X_train_img.std(axis=0, keepdims=True) + 1e-6
    X_train_scaled = (X_train_img - mean) / std
    X_test_scaled = None if X_test_img is None else (X_test_img - mean) / std

    class CNNAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder_layers: list[Any] = []
            current_channels = channels
            for out_channels in conv_channels:
                encoder_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
                encoder_layers.append(nn.ReLU())
                current_channels = out_channels
            self.encoder_conv = nn.Sequential(*encoder_layers)

            with torch.no_grad():
                sample = torch.zeros(1, channels, height, width)
                conv_output = self.encoder_conv(sample)
            self.conv_shape = tuple(conv_output.shape[1:])
            flattened_dim = int(conv_output[0].numel())

            self.encoder_head = nn.Linear(flattened_dim, latent_dim)
            self.decoder_head = nn.Linear(latent_dim, flattened_dim)

            decoder_layers: list[Any] = []
            reversed_channels = list(reversed(conv_channels))
            current_channels = reversed_channels[0]
            for next_channels in reversed_channels[1:]:
                decoder_layers.append(nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1))
                decoder_layers.append(nn.ReLU())
                current_channels = next_channels
            decoder_layers.append(nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1))
            self.decoder_conv = nn.Sequential(*decoder_layers)

        def forward(self, batch: Any) -> tuple[Any, Any]:
            conv_features = self.encoder_conv(batch)
            flattened = conv_features.flatten(start_dim=1)
            latent = self.encoder_head(flattened)
            decoded = self.decoder_head(latent).view(batch.shape[0], *self.conv_shape)
            reconstruction = self.decoder_conv(decoded)
            reconstruction = reconstruction[:, :, :height, :width]
            return latent, reconstruction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = data_utils.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

    def encode(array: Any) -> Any:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(array, dtype=torch.float32, device=device)
            latent, _ = model(tensor)
            return latent.cpu().numpy()

    train_embedding = encode(X_train_scaled)
    test_embedding = None if X_test_scaled is None else encode(X_test_scaled)
    return train_embedding, test_embedding


def fit_mlp_multilabel(
    method_spec: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
) -> dict[str, Any]:
    """Train a plain MLP baseline for multi-label prediction."""
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = np.asarray(to_dense_array(X_test), dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    hidden_dims = list(params.get("hidden_dims", [256, 128]))
    epochs = int(params.get("epochs", 25))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))

    class MultiLabelMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            layers: list[Any] = []
            current = input_dim
            for width in hidden_dims:
                layers.append(nn.Linear(current, width))
                layers.append(nn.ReLU())
                current = width
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(current, output_dim)

        def forward(self, batch: Any) -> tuple[Any, Any]:
            features = self.backbone(batch)
            logits = self.head(features)
            return features, logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelMLP(X_train_np.shape[1], y_train_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

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
            _, logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_features = torch.tensor(X_test_np, dtype=torch.float32, device=device)
        embeddings, logits = model(test_features)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        embedding_np = embeddings.cpu().numpy()

    return {"probabilities": probabilities, "embeddings": embedding_np}


def fit_cnn_multilabel(
    method_spec: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
) -> dict[str, Any]:
    """Train a CNN baseline for multi-label prediction on flattened image inputs."""
    np = import_required("numpy")
    torch = import_required("torch")
    nn = import_required("torch.nn")
    data_utils = import_required("torch.utils.data")

    params = method_spec.get("params", {})
    X_train_np = np.asarray(to_dense_array(X_train), dtype=np.float32)
    X_test_np = np.asarray(to_dense_array(X_test), dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    channels, height, width = _infer_image_shape(X_train_np.shape[1], params)
    conv_channels = list(params.get("conv_channels", [16, 32]))
    hidden_dim = int(params.get("hidden_dim", 128))
    epochs = int(params.get("epochs", 25))
    batch_size = int(params.get("batch_size", 128))
    learning_rate = float(params.get("learning_rate", 1e-3))

    X_train_img = _reshape_flat_images(X_train_np, channels, height, width)
    X_test_img = _reshape_flat_images(X_test_np, channels, height, width)

    class MultiLabelCNN(nn.Module):
        def __init__(self, output_dim: int) -> None:
            super().__init__()
            conv_layers: list[Any] = []
            current_channels = channels
            for out_channels in conv_channels:
                conv_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
                conv_layers.append(nn.ReLU())
                current_channels = out_channels
            self.encoder = nn.Sequential(*conv_layers)

            with torch.no_grad():
                sample = torch.zeros(1, channels, height, width)
                conv_output = self.encoder(sample)
            flattened_dim = int(conv_output[0].numel())

            self.embedding_head = nn.Linear(flattened_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, batch: Any) -> tuple[Any, Any]:
            encoded = self.encoder(batch)
            flattened = encoded.flatten(start_dim=1)
            embeddings = self.embedding_head(flattened)
            logits = self.classifier(embeddings)
            return embeddings, logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelCNN(y_train_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = data_utils.TensorDataset(
        torch.tensor(X_train_img, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            _, logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_features = torch.tensor(X_test_img, dtype=torch.float32, device=device)
        embeddings, logits = model(test_features)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        embedding_np = embeddings.cpu().numpy()

    return {"probabilities": probabilities, "embeddings": embedding_np}


def _infer_image_shape(flat_dim: int, params: dict[str, Any]) -> tuple[int, int, int]:
    input_shape = params.get("input_shape")
    if input_shape is not None:
        if len(input_shape) == 2:
            channels = 1
            height, width = (int(input_shape[0]), int(input_shape[1]))
            if channels * height * width != flat_dim:
                raise ValueError("Configured input_shape does not match the flattened feature dimension")
            return channels, height, width
        if len(input_shape) == 3:
            channels, height, width = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
            if channels * height * width != flat_dim:
                raise ValueError("Configured input_shape does not match the flattened feature dimension")
            return channels, height, width
        raise ValueError("input_shape must have length 2 or 3")

    side = int(flat_dim ** 0.5)
    if side * side != flat_dim:
        raise ValueError("CNN methods require square image-like inputs or an explicit input_shape")
    return 1, side, side


def _reshape_flat_images(array: Any, channels: int, height: int, width: int) -> Any:
    np = import_required("numpy")
    reshaped = np.asarray(array, dtype=np.float32)
    expected = channels * height * width
    if reshaped.shape[1] != expected:
        raise ValueError("Input feature dimension does not match the requested image shape")
    return reshaped.reshape(-1, channels, height, width)
