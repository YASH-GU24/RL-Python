//import com.google.protobuf.gradle.id
import com.google.protobuf.gradle.*


plugins {
    id("java")
    id("com.google.protobuf") version "0.9.4"
    kotlin("jvm") version "1.6.10"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    // https://mvnrepository.com/artifact/org.cloudsimplus/cloudsim-plus
    implementation("org.cloudsimplus:cloudsim-plus:6.2.7")
    // https://mvnrepository.com/artifact/org.knowm.xchart/xchart
    implementation("org.knowm.xchart:xchart:3.8.5")
    implementation(files("${projectDir}/src/main/java/libs/jFuzzyLogic_v3.0.jar"))
    // https://mvnrepository.com/artifact/io.grpc/grpc-protobuf
    implementation("io.grpc:grpc-protobuf:1.57.1")
    implementation("io.grpc:grpc-stub:1.57.1")
    implementation("io.grpc:grpc-all:1.57.1")
    implementation("io.grpc:grpc-kotlin-stub:1.3.0")
    implementation("com.google.protobuf:protobuf-java:3.6.1")
}



//protoc --plugin=protoc-gen-grpc-java=C:\protoc-24.0-win64\bin\protoc-gen-grpc-java-1.57.1-windows-x86_32.exe --grpc-java_out="build/generated/source/proto/main/java" "format.proto" --java_out="build/generated/source/proto/main/java" --proto_path="src/main/proto"
// Look here for set path
//sourceSets {
//    main {
//        proto {
//            srcDir("src/main/proto")
//        }
//    }
//}

protobuf {
    protoc { artifact = "com.google.protobuf:protoc:3.12.0" }
    plugins {
        id("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.57.1"
        }
    }
    generateProtoTasks {
        ofSourceSet("main").forEach {
            it.plugins {
                id("grpc")
            }
        }
    }
}