plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.retrofrost.malirvc"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.retrofrost.malirvc"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    packaging {
        resources.excludes += setOf("META-INF/INDEX.LIST", "META-INF/DEPENDENCIES")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // RVC synthesizer. The .pte bundled with the APK is a generic graph whose
    // weights are supplied at runtime from the user's .pth checkpoint.
    implementation("org.pytorch:executorch-android-vulkan:1.4.0")

    // ContentVec + RMVPE helper models. They are bundled as ONNX assets and run
    // offline; the RVC synthesizer itself is delegated to Vulkan for Mali.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
}
