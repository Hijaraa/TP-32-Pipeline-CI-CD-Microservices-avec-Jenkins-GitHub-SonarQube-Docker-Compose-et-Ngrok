# Pipeline CI/CD pour Microservices

Ce projet implémente un pipeline CI/CD complet utilisant Jenkins, SonarQube, Docker Compose, et GitHub pour automatiser le build, l'analyse de code, et le déploiement de microservices Spring Boot.

## 📋 Architecture du Projet

Le projet contient 4 microservices :
- **car** : Service de gestion des voitures
- **client** : Service de gestion des clients
- **gateway** : Service Gateway (Spring Cloud Gateway)
- **server_eureka** : Service Discovery (Netflix Eureka)

## 🛠️ Technologies Utilisées

- **Jenkins** : Automatisation CI/CD
- **SonarQube** : Analyse de qualité du code
- **Docker & Docker Compose** : Conteneurisation et orchestration
- **Maven** : Gestion des dépendances et build
- **Spring Boot 3.2.0** : Framework Java
- **MySQL** : Base de données
- **Consul** : Service Discovery
- **Ngrok** : Tunnel pour webhooks GitHub

## 📁 Structure du Projet

```
.
├── car/                    # Microservice Car
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/
├── client/                 # Microservice Client
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/
├── gateway/                # Microservice Gateway
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/
├── server_eureka/          # Service Discovery
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/
├── deploy/                 # Configuration de déploiement
│   └── docker-compose.yml
└── Jenkinsfile             # Pipeline Jenkins
```

## 🚀 Configuration du Pipeline CI/CD

### Prérequis

1. **Jenkins** installé et configuré
2. **SonarQube** accessible sur `http://localhost:9999`
3. **Docker** et **Docker Compose** installés
4. **Maven** installé
5. **Ngrok** configuré pour exposer Jenkins (pour les webhooks GitHub)

### Configuration Jenkins

#### 1. Installation des Plugins Requis

Dans Jenkins, installer les plugins suivants :
- **Pipeline**
- **Git**
- **Docker Pipeline**
- **SonarQube Scanner**

#### 2. Configuration des Credentials

1. **Token SonarQube** :
   - Aller dans `Jenkins` → `Manage Jenkins` → `Credentials`
   - Ajouter une credential de type "Secret text"
   - ID : `sonar-token`
   - Secret : Token généré depuis SonarQube (User → My Account → Security)

2. **Credentials GitHub** (si nécessaire) :
   - Ajouter les credentials pour accéder au repository GitHub

#### 3. Configuration SonarQube dans Jenkins

1. Aller dans `Jenkins` → `Manage Jenkins` → `Configure System`
2. Section "SonarQube servers" :
   - Ajouter un serveur SonarQube
   - Name : `SonarQube`
   - Server URL : `http://localhost:9999`
   - Server authentication token : Utiliser le token SonarQube

#### 4. Création du Job Jenkins

1. Créer un nouveau job de type **Pipeline**
2. Dans la configuration :
   - **Definition** : Pipeline script from SCM
   - **SCM** : Git
   - **Repository URL** : URL de votre repository GitHub
   - **Branch** : `main` ou `master`
   - **Script Path** : `Jenkinsfile`

#### 5. Configuration du Webhook GitHub

1. **Obtenir l'URL Ngrok** :
   ```bash
   ngrok http 8080  # Port par défaut de Jenkins
   ```
   Notez l'URL HTTPS fournie par Ngrok (ex: `https://xxxx.ngrok.io`)

2. **Configurer le Webhook dans GitHub** :
   - Aller dans votre repository GitHub → `Settings` → `Webhooks`
   - Cliquer sur `Add webhook`
   - **Payload URL** : `https://xxxx.ngrok.io/github-webhook/`
   - **Content type** : `application/json`
   - **Events** : Sélectionner "Just the push event"
   - Cliquer sur `Add webhook`

3. **Activer le trigger dans Jenkins** :
   - Dans la configuration du job Jenkins
   - Section "Build Triggers"
   - Cocher "GitHub hook trigger for GITScm polling"

## 🔄 Étapes du Pipeline

Le pipeline Jenkins exécute les étapes suivantes :

1. **Clonage** : Checkout de la branche main/master
2. **Build Maven - Car** : Compilation et packaging du service Car
3. **Build Maven - Client** : Compilation et packaging du service Client
4. **Build Maven - Gateway** : Compilation et packaging du service Gateway
5. **Build Maven - Server Eureka** : Compilation et packaging du service Eureka
6. **Analyse SonarQube - Car** : Analyse de qualité du code pour Car
7. **Analyse SonarQube - Client** : Analyse de qualité du code pour Client
8. **Docker Compose** : Build et déploiement des conteneurs

## ✅ Vérification du Fonctionnement

### 7.1 Lancer un Build Manuel

1. Dans Jenkins, ouvrir le job
2. Cliquer sur **Build Now**
3. **Résultat attendu** : Une exécution apparaît dans l'historique avec un console log accessible

### 7.2 Vérifier le Résultat dans Jenkins

Ouvrir **Console Output** et contrôler :

- ✅ **Stage clonage** : `checkout main` exécuté avec succès
- ✅ **Builds Maven** : Succès sur `car`, `client`, `gateway`, `server_eureka`
- ✅ **SonarQube** : Exécution `sonar:sonar` sur `car` et `client`
- ✅ **Docker Compose** : `up -d --build` exécuté avec succès

**Remarque** : Si un stage est rouge, lire la première erreur dans la console. Problèmes courants :
- Erreur de chemin Maven
- Token SonarQube invalide ou expiré
- Service Docker non démarré
- Port déjà utilisé

### 7.3 Vérifier les Tableaux de Bord SonarQube

1. Aller sur SonarQube : `http://localhost:9999`
2. Ouvrir le projet **car** → Vérifier qu'une analyse récente existe
3. Ouvrir le projet **client** → Vérifier qu'une analyse récente existe

**Résultat attendu** :
- Métriques affichées (bugs, vulnérabilités, code smells)
- "Last analysis" récent
- Dashboard avec les métriques de qualité

### 7.4 Vérifier le Déploiement Docker Compose

Sur la machine hôte, exécuter :

```bash
docker ps
```

**Résultat attendu** : Conteneurs démarrés (au minimum les services déployés par le compose du dossier `deploy/`) :
- `mysql-container1`
- `consul-container`
- `eureka-server`
- `gateway-service`
- `client-service`
- `voiture-service`
- `phpmyadmin-container`

#### Tester les Services (Optionnel)

Adapter les ports réels du `deploy/docker-compose.yml` :

```bash
# Exemple générique (à adapter selon vos ports)
curl http://localhost:8888/actuator/health  # Gateway
curl http://localhost:8089/actuator/health  # Car
curl http://localhost:8088/actuator/health  # Client
```

### 7.5 Tester le Déclenchement Automatique via Push GitHub

1. Faire une petite modification (ex. README) :
   ```bash
   git add README.md
   git commit -m "test: déclenchement webhook"
   git push
   ```

2. **Résultat attendu** : Jenkins démarre automatiquement une nouvelle exécution après le push

3. **Vérification** :
   - Aller dans Jenkins
   - Vérifier l'historique des builds
   - Un nouveau build doit apparaître avec le message de commit

**Si rien ne se lance, contrôler** :
- URL Ngrok actuelle (elle change à chaque redémarrage)
- Webhook GitHub actif (vérifier dans GitHub → Settings → Webhooks)
- Trigger Jenkins coché (Build Triggers → GitHub hook trigger)

## 🔧 Configuration des Variables d'Environnement

Le fichier `deploy/docker-compose.yml` utilise des variables d'environnement. Créer un fichier `.env` dans le dossier `deploy/` :

```env
# MySQL
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE_CLIENT=clientdb
MYSQL_PORT=3307

# Consul
CONSUL_PORT=8500
CONSUL_HOST=consul

# Eureka
EUREKA_PORT=8761

# Gateway
GATEWAY_PORT=8888

# Client
CLIENT_PORT=8088
SPRING_DATASOURCE_URL_CLIENT=jdbc:mysql://mysql:3306/clientdb
SPRING_DATASOURCE_USERNAME=root
SPRING_DATASOURCE_PASSWORD=rootpassword

# Car
CAR_PORT=8089
SPRING_DATASOURCE_URL_CAR=jdbc:mysql://mysql:3306/cardb
```

## 📊 Ports des Services

| Service | Port | URL |
|---------|------|-----|
| Gateway | 8888 | http://localhost:8888 |
| Client | 8088 | http://localhost:8088 |
| Car | 8089 | http://localhost:8089 |
| Eureka | 8761 | http://localhost:8761 |
| Consul | 8500 | http://localhost:8500 |
| MySQL | 3307 | localhost:3307 |
| phpMyAdmin | 8081 | http://localhost:8081 |
| SonarQube | 9999 | http://localhost:9999 |

## 🐛 Dépannage

### Problème : Build Maven échoue

- Vérifier que Maven est installé : `mvn --version`
- Vérifier les dépendances dans les `pom.xml`
- Nettoyer le cache Maven : `mvn clean`

### Problème : SonarQube ne fonctionne pas

- Vérifier que SonarQube est démarré : `http://localhost:9999`
- Vérifier le token SonarQube dans les credentials Jenkins
- Vérifier l'URL SonarQube dans la configuration Jenkins

### Problème : Docker Compose échoue

- Vérifier que Docker est démarré : `docker ps`
- Vérifier les ports disponibles
- Vérifier les variables d'environnement dans `.env`

### Problème : Webhook GitHub ne déclenche pas Jenkins

- Vérifier l'URL Ngrok (elle change à chaque redémarrage)
- Mettre à jour le webhook GitHub avec la nouvelle URL
- Vérifier que le trigger est activé dans Jenkins
- Vérifier les logs Jenkins pour les erreurs de webhook

## 📝 Notes Importantes

- Le pipeline exécute `mvn clean package -DskipTests` pour accélérer le build
- Seuls les services `car` et `client` sont analysés par SonarQube
- Le pipeline utilise `docker-compose down` avant `up` pour éviter les conflits
- Les tokens et credentials doivent être configurés dans Jenkins avant le premier build

## 🔗 Liens Utiles

- [Documentation Jenkins](https://www.jenkins.io/doc/)
- [Documentation SonarQube](https://docs.sonarqube.org/)
- [Documentation Docker Compose](https://docs.docker.com/compose/)
- [Documentation Spring Boot](https://spring.io/projects/spring-boot)

## 📄 Licence

Ce projet est fourni à des fins éducatives.

---

**Auteur** : Équipe de développement  
**Date** : 2024

