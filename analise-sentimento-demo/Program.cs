using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;
using System.Collections.Generic;

namespace analise_sentimento_demo
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        public static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var splitDataView = LoadData(mlContext);

            var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseLoadedModelWithBatchItems(mlContext);

            Console.WriteLine();
            Console.WriteLine("Fim!");
            Console.ReadKey();
        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            var splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            Console.WriteLine(">>> Treinamento do modelo");

            var model = pipeline.Fit(splitTrainSet);

            Console.WriteLine(">>>>> Fim treinamento do modelo");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine(">>> Analisando acurácia do modelo com teste de dados");
            var predictions = model.Transform(splitTestSet);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine(">>> Avaliação de métricas de qualidade do modelo");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Acurácia: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc (quanto maior o AUC, melhor o modelo é para distinguir entre duas classes): {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine(">>> Fim avaliação de métricas de qualidade");

            SaveModelAsFile(mlContext, model);
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("Modelo salvo. Diretório: {0}", _modelPath);
        }

        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "I didn't like this outfit."
                },
                new SentimentData
                {
                    SentimentText = "I want so much a new computer!"
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read)) { loadedModel = mlContext.Model.Load(stream); }

            var sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);

            var predictions = loadedModel.Transform(sentimentStreamingDataView);
            
            var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine(">>> Teste de predição com múltiplas amostras");
            Console.WriteLine();
            
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            foreach ((SentimentData sentiment, SentimentPrediction prediction) item in sentimentsAndPredictions)
            {
                Console.WriteLine(
                    $"Sentimento: {item.sentiment.SentimentText} | " +
                    $"Predição: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positivo" : "Negativo")} | " +
                    $"Probabilidade: {item.prediction.Probability} ");
            }

            Console.WriteLine();
            Console.WriteLine(">>> Fim das predições");    
        }
    }
}
