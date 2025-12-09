import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [currentPage, setCurrentPage] = useState('home');
  const [menuOpen, setMenuOpen] = useState(false);
  const [navbarVisible, setNavbarVisible] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  // Custom cursor effect
  useEffect(() => {
    const handleMouseMove = (e) => {
      setCursorPos({ x: e.clientX, y: e.clientY });
      
      // Create trail effect
      const trail = document.createElement('div');
      trail.className = 'cursor-trail';
      trail.style.left = e.clientX + 'px';
      trail.style.top = e.clientY + 'px';
      document.body.appendChild(trail);
      
      setTimeout(() => {
        trail.remove();
      }, 500);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Scroll navbar hide/show
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      
      if (currentScrollY < lastScrollY || currentScrollY < 100) {
        // Scrolling up or at top
        setNavbarVisible(true);
      } else if (currentScrollY > lastScrollY && currentScrollY > 100) {
        // Scrolling down
        setNavbarVisible(false);
      }
      
      setLastScrollY(currentScrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [lastScrollY]);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    } else {
      alert('Silakan pilih file gambar yang valid!');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
      } else {
        alert('Silakan pilih file gambar yang valid!');
      }
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setResult(null);
    setScanProgress(0);

    // Animasi scan progress
    const progressInterval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return 95;
        }
        return prev + 5;
      });
    }, 100);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      clearInterval(progressInterval);
      setScanProgress(100);
      
      setTimeout(() => {
        setResult(response.data);
        setIsLoading(false);
      }, 500);
    } catch (error) {
      clearInterval(progressInterval);
      console.error('Error:', error);
      setResult({
        success: false,
        message: 'Terjadi kesalahan',
        detail: error.response?.data?.error || 'Tidak dapat terhubung ke server'
      });
      setIsLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setScanProgress(0);
  };

  const navigateTo = (page) => {
    setCurrentPage(page);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToSection = (sectionId) => {
    setMenuOpen(false);
    const element = document.getElementById(sectionId);
    if (element) {
      const offset = 100; // navbar height
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  // Render Home Page
  const renderHomePage = () => (
    <>
      {/* Hero Section */}
      <div className="hero">
        <div className="hero-content">
          <div className="logo-container">
            <div className="logo-circle">
              <span className="logo-icon">🐟</span>
            </div>
          </div>
          <h1 className="hero-title">
            <span className="gradient-text">Fish AI</span> Classifier
          </h1>
          <p className="hero-subtitle">
            Teknologi AI canggih untuk mengidentifikasi lebih dari 100 spesies ikan
          </p>
          <p className="hero-description">
            Menggunakan arsitektur <strong>ResNet50</strong> yang telah dilatih dengan ribuan gambar ikan dari berbagai spesies. 
            Sistem kami dapat mengidentifikasi ikan dengan tingkat akurasi hingga <strong>95%</strong> hanya dalam waktu kurang dari 2 detik!
          </p>
          <div className="stats">
            <div className="stat-item">
              <div className="stat-number">12+</div>
              <div className="stat-label">Spesies</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">95%</div>
              <div className="stat-label">Akurasi</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">{"< 2s"}</div>
              <div className="stat-label">Kecepatan</div>
            </div>
          </div>
        </div>
        <div className="waves">
          <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="wave">
            <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" opacity=".25" className="wave-path"></path>
            <path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z" opacity=".5" className="wave-path"></path>
            <path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" className="wave-path"></path>
          </svg>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="container">
          
          {/* How it Works Section */}
          <div className="info-section" id="cara-kerja">
            <h2 className="section-title">Bagaimana Cara Kerjanya?</h2>
            <p className="section-description">
              Sistem klasifikasi ikan kami menggunakan teknologi <strong>Deep Learning</strong> dengan model <strong>ResNet50</strong> 
              yang telah dioptimalkan khusus untuk mengenali berbagai spesies ikan air tawar dan laut. 
              Prosesnya sangat sederhana dan cepat!
            </p>
            <div className="steps">
              <div className="step-card">
                <div className="step-icon">📤</div>
                <h3 className="step-title">1. Upload Gambar</h3>
                <p className="step-desc">
                  Pilih foto ikan dari galeri Anda atau ambil foto langsung. Sistem kami mendukung format JPG, PNG, dan JPEG.
                  Pastikan gambar ikan terlihat jelas untuk hasil terbaik.
                </p>
              </div>
              <div className="step-card">
                <div className="step-icon">🔍</div>
                <h3 className="step-title">2. AI Menganalisis</h3>
                <p className="step-desc">
                  Model ResNet50 kami akan memproses gambar melalui 50 lapisan neural network. 
                  AI mengekstrak fitur unik seperti bentuk sirip, pola sisik, dan warna untuk identifikasi akurat.
                </p>
              </div>
              <div className="step-card">
                <div className="step-icon">✨</div>
                <h3 className="step-title">3. Dapatkan Hasil</h3>
                <p className="step-desc">
                  Dalam hitungan detik, Anda akan melihat nama spesies ikan, tingkat keyakinan AI, 
                  dan 5 prediksi teratas lainnya. Lengkap dengan nama ilmiah dan rekomendasi!
                </p>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div className="features-section">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3 className="feature-title">Akurasi Tinggi</h3>
              <p className="feature-desc">
                Model ResNet50 terlatih dengan ribuan gambar berkualitas tinggi. 
                Menggunakan teknik transfer learning dan fine-tuning untuk mencapai akurasi maksimal.
              </p>
              <span className="feature-badge">95% Akurat</span>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3 className="feature-title">Proses Cepat</h3>
              <p className="feature-desc">
                Optimasi backend dengan TensorFlow dan caching pintar memastikan hasil identifikasi 
                dalam waktu kurang dari 2 detik, bahkan untuk gambar beresolusi tinggi.
              </p>
              <span className="feature-badge">{"< 2 Detik"}</span>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🌊</div>
              <h3 className="feature-title">Multi Spesies</h3>
              <p className="feature-desc">
                Database lengkap mencakup lebih dari 12 spesies ikan dari berbagai habitat: 
                air tawar, air laut, ikan hias, hingga ikan konsumsi populer.
              </p>
              <span className="feature-badge">12+ Spesies</span>
            </div>
          </div>

          {/* Classifier Section */}
          <div className="classifier-section" id="klasifikasi">
            <div className="classifier-header">
              <h2 className="classifier-title">🔬 Coba Klasifikasi Ikan</h2>
              <p className="classifier-subtitle">
                Upload gambar ikan untuk mengetahui jenisnya secara instan dengan teknologi AI
              </p>
              <div className="classifier-note">
                <strong>💡 Tips:</strong> Gunakan foto dengan pencahayaan baik, fokus jelas, dan ikan terlihat utuh untuk hasil optimal. 
                Format yang didukung: JPG, PNG, JPEG (Maks. 10MB)
              </div>
            </div>

            {!previewUrl && (
              <div 
                className={`upload-zone ${dragActive ? 'dragover' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById('fileInput').click()}
              >
                <div className="upload-icon">
                  <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <h3 className="upload-title">Upload Gambar Ikan</h3>
                <p className="upload-text">Drag & drop atau klik untuk memilih</p>
                <p className="upload-hint">Mendukung: JPG, PNG, JPEG</p>
                <input
                  id="fileInput"
                  type="file"
                  className="file-input"
                  accept="image/*"
                  onChange={handleFileSelect}
                />
                <button className="upload-btn">
                  Pilih Gambar
                </button>
              </div>
            )}

            {previewUrl && !result && (
              <div className="preview-section">
                <div className="image-container">
                  <img src={previewUrl} alt="Preview" className="preview-image" />
                  {isLoading && (
                    <div className="scan-overlay">
                      <div className="scan-line" />
                      <div className="scan-corners">
                        <div className="scan-corner top-left" />
                        <div className="scan-corner top-right" />
                        <div className="scan-corner bottom-left" />
                        <div className="scan-corner bottom-right" />
                      </div>
                      <div className="scan-progress">
                        <div className="scan-progress-text">Scanning... {scanProgress}%</div>
                        <div className="progress-bar">
                          <div className="progress-fill" style={{ width: `${scanProgress}%` }} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div className="action-buttons">
                  <button 
                    className="btn btn-primary"
                    onClick={analyzeImage}
                    disabled={isLoading}
                  >
                    {isLoading ? '🔄 Menganalisis...' : '🔍 Analisis Sekarang'}
                  </button>
                  <button 
                    className="btn btn-secondary"
                    onClick={resetApp}
                    disabled={isLoading}
                  >
                    ❌ Batal
                  </button>
                </div>
              </div>
            )}

            {result && result.success && (
              <div className={`result-card ${result.prediction === 'Bukan Ikan' ? 'not-fish' : ''}`}>
                <div className="result-header">
                  <div className="result-icon">
                    {result.prediction === 'Bukan Ikan' ? '⛔' : '✅'}
                  </div>
                  <h2 className="result-title">
                    {result.prediction === 'Bukan Ikan' ? 'Bukan Ikan Terdeteksi' : 'Ikan Teridentifikasi!'}
                  </h2>
                </div>
                
                <div className="fish-name">
                  {result.prediction !== 'Bukan Ikan' && result.certainty_emoji} {result.prediction}
                </div>
                
                {result.prediction_en && result.prediction_en !== result.prediction && (
                  <div className="fish-name-en">({result.prediction_en})</div>
                )}
                
                <div className="confidence">
                  <div className="confidence-label">Tingkat Keyakinan</div>
                  <div className="confidence-value">{result.confidence_percentage || result.percentage}</div>
                  <div className="confidence-badge">{result.certainty_text || (result.confidence >= 0.8 ? 'Sangat Yakin' : result.confidence >= 0.6 ? 'Cukup Yakin' : 'Kurang Yakin')}</div>
                </div>

                {result.analysis && (
                  <div className="analysis-details">
                    <div className="analysis-title">🔬 Detail Analisis</div>
                    <div className="analysis-grid">
                      <div className="analysis-item">
                        <span className="analysis-icon">🎨</span>
                        <span className="analysis-text">{result.analysis.color}</span>
                      </div>
                      <div className="analysis-item">
                        <span className="analysis-icon">📐</span>
                        <span className="analysis-text">{result.analysis.shape}</span>
                      </div>
                      <div className="analysis-item">
                        <span className="analysis-icon">🧮</span>
                        <span className="analysis-text">{result.analysis.features_analyzed}</span>
                      </div>
                      <div className="analysis-item">
                        <span className="analysis-icon">🎯</span>
                        <span className="analysis-text">Model Accuracy: {result.analysis.model_accuracy}</span>
                      </div>
                    </div>
                  </div>
                )}

                {result.top_3_predictions && (
                  <div className="top-predictions">
                    <h3 className="top-title">🏆 Top 3 Prediksi Alternatif</h3>
                    {result.top_3_predictions.slice(0, 3).map((pred, index) => (
                      <div key={index} className="prediction-item">
                        <span className="pred-rank">{index + 1}</span>
                        <span className="pred-name">{pred.class_id || pred.class}</span>
                        <span className="pred-percent">{pred.confidence_percentage || `${(pred.confidence * 100).toFixed(2)}%`}</span>
                      </div>
                    ))}
                  </div>
                )}

                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="recommendations">
                    <div className="rec-title">💡 Rekomendasi</div>
                    <ul className="rec-list">
                      {result.recommendations.map((rec, index) => (
                        <li key={index} className="rec-item">{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.top_5_predictions && (
                  <div className="top-predictions">
                    <h3 className="top-title">Top 5 Prediksi</h3>
                    {result.top_5_predictions.map((pred, index) => (
                      <div key={index} className="prediction-item">
                        <span className="pred-rank">{index + 1}</span>
                        <span className="pred-name">{pred.name_id}</span>
                        <span className="pred-percent">{pred.percentage}</span>
                      </div>
                    ))}
                  </div>
                )}

                <button className="btn btn-try-again" onClick={resetApp}>
                  🔄 Coba Gambar Lain
                </button>
              </div>
            )}

            {result && !result.success && (
              <div className="result-card result-error">
                <div className="error-icon">❌</div>
                <h2 className="error-title">{result.message}</h2>
                <p className="error-detail">{result.detail}</p>
                {result.confidence && (
                  <p className="error-confidence">
                    Tingkat keyakinan: {result.confidence}% (minimum {result.threshold}%)
                  </p>
                )}
                <button className="btn btn-try-again" onClick={resetApp}>
                  🔄 Coba Lagi
                </button>
              </div>
            )}
          </div>

        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3 className="footer-title">🐟 Fish AI Classifier</h3>
            <p className="footer-text">
              Platform identifikasi ikan berbasis kecerdasan buatan yang membantu Anda mengenali 
              berbagai spesies ikan dengan cepat dan akurat.
            </p>
            <p className="footer-text" style={{ marginTop: '15px' }}>
              Cocok untuk peneliti, nelayan, pembudidaya ikan, dan pecinta akuarium.
            </p>
          </div>
          <div className="footer-section">
            <h4 className="footer-subtitle">⚙️ Teknologi</h4>
            <p className="footer-text">• ResNet50 Architecture</p>
            <p className="footer-text">• TensorFlow 2.x</p>
            <p className="footer-text">• Deep Learning CNN</p>
            <p className="footer-text">• Transfer Learning</p>
            <p className="footer-text">• Image Recognition AI</p>
          </div>
          <div className="footer-section">
            <h4 className="footer-subtitle">ℹ️ Informasi</h4>
            <p className="footer-text">📚 Tugas Kecerdasan Buatan</p>
            <p className="footer-text">🎓 Computer Vision Project</p>
            <p className="footer-text">🔬 Machine Learning Research</p>
            <p className="footer-text" style={{ marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '15px' }}>
              © 2025 Fish AI Classifier<br/>All Rights Reserved
            </p>
          </div>
        </div>
      </footer>
    </>
  );

  // Render Team Page
  const renderTeamPage = () => (
    <>
      <div className="main-content">
        <div className="container">
          <div className="team-section">
            <div className="team-intro">
              <h2 className="section-title">Tentang Tim</h2>
              <p className="team-description">
                Kami adalah mahasiswa yang bersemangat dalam bidang Kecerdasan Buatan dan Computer Vision. 
                Proyek Fish AI Classifier ini merupakan hasil kolaborasi tim dalam menerapkan teknologi 
                Deep Learning untuk memecahkan masalah identifikasi spesies ikan.
              </p>
            </div>

            <div className="team-grid">
              <div className="team-card">
                <div className="team-photo">
                  <div className="team-photo-placeholder">
                    {/* Ganti dengan gambar: <img src="/path/to/photo1.jpg" alt="Anggota 1" /> */}
                    <span className="team-icon">👨‍💻</span>
                  </div>
                </div>
                <div className="team-info">
                  <h3 className="team-name">Nama Anggota 1</h3>
                  <p className="team-role">AI Developer</p>
                  <p className="team-id">NIM: 123456789</p>
                  <p className="team-desc">
                    Bertanggung jawab dalam pengembangan model ResNet50 dan training dataset
                  </p>
                </div>
              </div>

              <div className="team-card">
                <div className="team-photo">
                  <div className="team-photo-placeholder">
                    {/* Ganti dengan gambar: <img src="/path/to/photo2.jpg" alt="Anggota 2" /> */}
                    <span className="team-icon">👨‍💻</span>
                  </div>
                </div>
                <div className="team-info">
                  <h3 className="team-name">Nama Anggota 2</h3>
                  <p className="team-role">Backend Developer</p>
                  <p className="team-id">NIM: 123456790</p>
                  <p className="team-desc">
                    Mengembangkan API backend dan integrasi model dengan sistem
                  </p>
                </div>
              </div>

              <div className="team-card">
                <div className="team-photo">
                  <div className="team-photo-placeholder">
                    {/* Ganti dengan gambar: <img src="/path/to/photo3.jpg" alt="Anggota 3" /> */}
                    <span className="team-icon">👨‍💻</span>
                  </div>
                </div>
                <div className="team-info">
                  <h3 className="team-name">Nama Anggota 3</h3>
                  <p className="team-role">Frontend Developer</p>
                  <p className="team-id">NIM: 123456791</p>
                  <p className="team-desc">
                    Merancang dan mengimplementasikan user interface yang interaktif
                  </p>
                </div>
              </div>
            </div>

            <div className="team-grid-single">
              <div className="team-card">
                <div className="team-photo">
                  <div className="team-photo-placeholder">
                    {/* Ganti dengan gambar: <img src="/path/to/photo4.jpg" alt="Anggota 4" /> */}
                    <span className="team-icon">👨‍💻</span>
                  </div>
                </div>
                <div className="team-info">
                  <h3 className="team-name">Nama Anggota 4</h3>
                  <p className="team-role">Data Scientist</p>
                  <p className="team-id">NIM: 123456792</p>
                  <p className="team-desc">
                    Melakukan preprocessing data dan evaluasi performa model
                  </p>
                </div>
              </div>
            </div>

            <div className="project-info-section">
              <h2 className="section-title">Informasi Proyek</h2>
              <div className="project-details">
                <div className="project-detail-card">
                  <div className="project-detail-icon">🎓</div>
                  <h3>Mata Kuliah</h3>
                  <p>Kecerdasan Buatan</p>
                </div>
                <div className="project-detail-card">
                  <div className="project-detail-icon">📅</div>
                  <h3>Tahun Akademik</h3>
                  <p>2024/2025</p>
                </div>
                <div className="project-detail-card">
                  <div className="project-detail-icon">🏫</div>
                  <h3>Universitas</h3>
                  <p>Universitas Maritim Raja Ali Haji</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3 className="footer-title">🐟 Fish AI Classifier</h3>
            <p className="footer-text">
              Platform identifikasi ikan berbasis kecerdasan buatan
            </p>
          </div>
          <div className="footer-section">
            <h4 className="footer-subtitle">⚙️ Teknologi</h4>
            <p className="footer-text">• ResNet50 Architecture</p>
            <p className="footer-text">• TensorFlow 2.x</p>
            <p className="footer-text">• Deep Learning CNN</p>
          </div>
          <div className="footer-section">
            <h4 className="footer-subtitle">ℹ️ Informasi</h4>
            <p className="footer-text">© 2025 Fish AI Classifier</p>
            <p className="footer-text">All Rights Reserved</p>
          </div>
        </div>
      </footer>
    </>
  );

  return (
    <div className="app">
      {/* Custom Cursor */}
      <div 
        className="custom-cursor" 
        style={{ 
          left: `${cursorPos.x - 20}px`, 
          top: `${cursorPos.y - 20}px` 
        }}
      >
        <span className="custom-cursor-icon">🐠</span>
      </div>

      {/* Navbar */}
      <nav className={`navbar ${navbarVisible ? 'visible' : 'hidden'}`}>
        <div className="navbar-container">
          <div className="navbar-logo" onClick={() => {
            setCurrentPage('home');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}>
            <span className="navbar-logo-icon">
              {/* Ganti dengan logo Anda - contoh: */}
              {/* <img src="/path/to/your/logo.png" alt="Logo" /> */}
              {/* Atau tetap pakai emoji: */}
              🐟
            </span>
            <span className="navbar-logo-text">Fish AI</span>
          </div>
          
          <button className="navbar-toggle" onClick={() => setMenuOpen(!menuOpen)}>
            <span></span>
            <span></span>
            <span></span>
          </button>

          <ul className={`navbar-menu ${menuOpen ? 'active' : ''}`}>
            <li className={`navbar-item ${currentPage === 'home' ? 'active' : ''}`}>
              <a onClick={() => {
                setCurrentPage('home');
                window.scrollTo({ top: 0, behavior: 'smooth' });
                setMenuOpen(false);
              }} className="navbar-link">
                🏠 Home
              </a>
            </li>
            <li className="navbar-item">
              <a onClick={() => {
                setCurrentPage('home');
                setTimeout(() => scrollToSection('cara-kerja'), 100);
              }} className="navbar-link">
                📚 Cara Kerja
              </a>
            </li>
            <li className="navbar-item">
              <a onClick={() => {
                setCurrentPage('home');
                setTimeout(() => scrollToSection('klasifikasi'), 100);
              }} className="navbar-link">
                🔬 Coba Klasifikasi
              </a>
            </li>
            <li className={`navbar-item ${currentPage === 'team' ? 'active' : ''}`}>
              <a onClick={() => navigateTo('team')} className="navbar-link">
                👥 Tim Kami
              </a>
            </li>
          </ul>
        </div>
      </nav>

      {/* Page Content */}
      {currentPage === 'home' && renderHomePage()}
      {currentPage === 'team' && renderTeamPage()}
    </div>
  );
}

export default App;