# ============================================================================
# 🐟 TRAINING MODEL IKAN - BELAJAR DARI SEMUA DATA 100%
# Model akan belajar dari SEMUA 1,459 gambar tanpa validation split
# ============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

print("="*70)
print("🐟 TRAINING MODEL IKAN - BELAJAR DARI SEMUA DATA")
print("="*70)
print("✨ Strategi:")
print("   1. Gunakan 100% data untuk training (1,459 gambar)")
print("   2. Tidak ada validation split")
print("   3. Model akan belajar MAKSIMAL dari semua data")
print("   4. Class weights untuk handle imbalance")
print("="*70 + "\n")

# ============================================================================
# KONFIGURASI
# ============================================================================

IMG_SIZE = 224           
BATCH_SIZE = 16          
EPOCHS = 100             # Lebih banyak untuk fine-tuning yang dalam
INITIAL_LR = 0.0001      # Learning rate lebih kecil untuk fine-tuning       

DATASET_PATH = './dataset'
OUTPUT_DIR = './models'

# Import untuk regularization
import tensorflow as tf

FISH_NAMES_ID = {
    'Black Sea Sprat': 'Ikan Sprat Laut Hitam',
    'Catfish': 'Ikan Lele',
    'Gilt Head Bream': 'Ikan Gilthead',
    'Horse Mackerel': 'Ikan Kembung',
    'Red Mullet': 'Ikan Kuniran',
    'Red Sea Bream': 'Ikan Kakap Merah',
    'Sea Bass': 'Ikan Kakap Putih',
    'Shrimp': 'Udang',
    'Striped Red Mullet': 'Ikan Kuniran Bergaris',
    'Trout': 'Ikan Trout',
    'Tilapia': 'Ikan Nila',
    'Ikan Nila': 'Ikan Nila',
    'Pufferfish': 'Ikan Buntal',
    'Grouper': 'Ikan Kerapu',
    'Kerapu': 'Ikan Kerapu',
    'Not Fish': 'Bukan Ikan'
}

# ============================================================================
# CEK DATASET
# ============================================================================
def check_dataset():
    print("="*70)
    print("📂 ANALISIS DATASET")
    print("="*70)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Folder tidak ditemukan!")
        return False

    folders = sorted([f for f in os.listdir(DATASET_PATH)
                     if os.path.isdir(os.path.join(DATASET_PATH, f))])

    if len(folders) == 0:
        print("❌ ERROR: Tidak ada subfolder!")
        return False

    print(f"✅ Dataset ditemukan! Total kelas: {len(folders)}\n")

    total_images = 0
    class_distribution = {}
    
    for folder in folders:
        path = os.path.join(DATASET_PATH, folder)
        images = [f for f in os.listdir(path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        class_distribution[folder] = count

        indo_name = FISH_NAMES_ID.get(folder, folder)
        status = "✅"
        print(f"{status} {folder:30s} ({indo_name:25s}): {count:4d} gambar")

    print(f"\n📊 TOTAL: {total_images} gambar")
    print(f"💡 SEMUA data ini akan digunakan untuk training (100%)!")
    print("="*70 + "\n")
    
    return True, class_distribution

# ============================================================================
# MODEL
# ============================================================================
def create_model(num_classes):
    print("🏗️  Membangun model MobileNetV2...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # UNFREEZE beberapa layer terakhir untuk fine-tuning
    # Ini membuat model BENAR-BENAR BELAJAR fitur ikan!
    base_model.trainable = True
    
    # Freeze layer awal (general features), unfreeze layer akhir (specific features)
    fine_tune_at = 100  # Unfreeze dari layer 100 ke atas
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"   ✅ MobileNetV2 loaded")
    print(f"   🔓 {trainable_layers} layers unlocked for fine-tuning (belajar fitur ikan!)")

    # Build model dengan lebih banyak layer untuk pembelajaran mendalam
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])

    print(f"\n📊 INFO MODEL:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    return model

# ============================================================================
# DATA GENERATOR - 100% UNTUK TRAINING
# ============================================================================
def create_data_generator():
    print("📊 Setup data generator (100% training)...")

    # Heavy augmentation karena tidak ada validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    print("   ✅ Heavy augmentation untuk generalisasi")
    print("   ✅ 100% data digunakan untuk training")

    return train_datagen

# ============================================================================
# TRAINING
# ============================================================================
def train_full_data():
    
    # Check dataset
    result = check_dataset()
    if not result:
        return None, None
    
    check_result, class_distribution = result

    print("="*70)
    print("🚀 MEMULAI TRAINING - 100% DATA")
    print("="*70)
    print("💡 Strategi:")
    print("   • Gunakan SEMUA 1,459 gambar untuk training")
    print("   • Heavy augmentation untuk mencegah overfitting")
    print("   • Class weights untuk handle imbalance")
    print("   • Lebih banyak epoch (80 epoch)")
    print("="*70 + "\n")

    # Setup generator
    train_datagen = create_data_generator()

    # Load ALL data untuk training (tanpa validation_split)
    print("📥 Loading ALL data untuk training...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    num_classes = len(train_gen.class_indices)

    print(f"\n✅ Data loaded!")
    print(f"   Training samples: {train_gen.samples} (100% data!)")
    print(f"   Classes: {num_classes}")
    print(f"   Batches per epoch: {len(train_gen)}")
    
    # Calculate class weights
    print(f"\n⚖️  Calculating class weights...")
    labels = []
    for i in range(min(100, len(train_gen))):
        _, batch_labels = train_gen[i]
        labels.extend(np.argmax(batch_labels, axis=1))
    
    unique_classes = np.unique(labels)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    class_weights = {i: class_weights_array[i] for i in range(len(unique_classes))}
    print(f"   ✅ Class weights calculated")
    
    # Save class mapping
    class_map = {}
    for name, idx in train_gen.class_indices.items():
        class_map[str(idx)] = {
            'name_en': name,
            'name_id': FISH_NAMES_ID.get(name, name)
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(OUTPUT_DIR, f'full_data_model_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    with open(f'{model_dir}/class_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    # Save dataset info
    dataset_info = {
        'total_images': train_gen.samples,
        'num_classes': num_classes,
        'class_distribution': class_distribution,
        'training_strategy': '100% data for training',
        'augmentation': 'Heavy (rotation, shift, zoom, flip)',
        'timestamp': timestamp
    }
    
    with open(f'{model_dir}/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)

    # Create model
    model = create_model(num_classes)

    # Compile
    print("\n🔧 Compile model...")
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
    )
    print("   ✅ Model compiled!\n")

    # Callbacks (tanpa EarlyStopping karena tidak ada validation)
    callbacks = [
        ModelCheckpoint(
            f'{model_dir}/model_checkpoint_epoch{{epoch:02d}}.h5',
            save_freq='epoch',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=8,
            verbose=1,
            min_lr=1e-7
        )
    ]

    # Training
    print("="*70)
    print("🏋️  MULAI TRAINING - MODEL BELAJAR DARI SEMUA DATA")
    print("="*70)
    print("📚 Model akan:")
    print("   ✓ Belajar dari SEMUA 1,459 gambar")
    print("   ✓ Heavy augmentation untuk generalisasi")
    print("   ✓ Fine-tuning deep layers (belajar fitur ikan!)")
    print("   ✓ Training selama 100 epoch dengan learning rate kecil")
    print("   ✓ Learning rate otomatis menyesuaikan")
    print("="*70 + "\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save final model
    model.save(f'{model_dir}/model_final.h5')
    print(f"\n✅ Model final saved!")

    # Save best model (epoch dengan loss terendah)
    best_epoch = np.argmin(history.history['loss'])
    print(f"\n🏆 Best epoch: {best_epoch + 1} (loss: {history.history['loss'][best_epoch]:.4f})")
    
    # Copy best checkpoint as model_best.h5
    import shutil
    best_checkpoint = f'{model_dir}/model_checkpoint_epoch{best_epoch+1:02d}.h5'
    if os.path.exists(best_checkpoint):
        shutil.copy(best_checkpoint, f'{model_dir}/model_best.h5')
        print(f"   ✅ Best model saved as model_best.h5")

    # Save history
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    history_dict['best_epoch'] = int(best_epoch + 1)
    
    with open(f'{model_dir}/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Final stats
    print("\n" + "="*70)
    print("📊 TRAINING SELESAI")
    print("="*70)
    
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_top3 = history.history['top_3_acc'][-1]
    
    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Final Training Accuracy: {final_acc*100:.2f}%")
    print(f"Final Top-3 Accuracy: {final_top3*100:.2f}%")
    
    best_loss = history.history['loss'][best_epoch]
    best_acc = history.history['accuracy'][best_epoch]
    
    print(f"\nBest Training Loss: {best_loss:.4f} (epoch {best_epoch + 1})")
    print(f"Best Training Accuracy: {best_acc*100:.2f}% (epoch {best_epoch + 1})")

    # Plot
    plot_history(history, model_dir)

    print("\n" + "="*70)
    print("🎉 TRAINING SELESAI!")
    print("="*70)
    print(f"📁 Model tersimpan: {model_dir}/")
    print("   ✅ model_best.h5 (epoch terbaik)")
    print("   ✅ model_final.h5 (epoch terakhir)")
    print("   ✅ class_mapping.json")
    print("   ✅ dataset_info.json")
    print("="*70)

    print("\n💡 Selanjutnya:")
    print(f"   1. cp {model_dir}/model_best.h5 models/")
    print(f"   2. cp {model_dir}/class_mapping.json models/")
    print("   3. Restart backend API")
    print("   4. Test dengan gambar ikan!")
    print("="*70)

    return model, history, model_dir

# ============================================================================
# PLOT
# ============================================================================
def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2, color='blue')
    best_epoch = np.argmax(history.history['accuracy'])
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: epoch {best_epoch+1}')
    axes[0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Training', linewidth=2, color='orange')
    best_epoch = np.argmin(history.history['loss'])
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: epoch {best_epoch+1}')
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🐟"*35)
    print("TRAINING DENGAN 100% DATA - MAKSIMAL LEARNING")
    print("🐟"*35 + "\n")

    model, history, model_dir = train_full_data()

    if model is not None:
        print("\n🎊 MODEL SUDAH BELAJAR DARI SEMUA DATA!")
        print("Model sekarang sudah menyerap semua pengetahuan dari 1,459 gambar!")
