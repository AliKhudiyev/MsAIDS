package example;

import org.apache.qpid.jms.JmsConnectionFactory;
import javax.jms.Connection;
import javax.jms.Session;
import javax.jms.Destination;
import javax.jms.MessageConsumer;
import javax.jms.TextMessage;
import javax.jms.*;

class Consumer {

    public static void main(String[] args) throws JMSException {

        JmsConnectionFactory factory = new JmsConnectionFactory("amqp://localhost:5672");
        Connection connection = factory.createConnection("admin", "password");
        connection.start();
        Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
        
        Destination destination = null;
        if(args.length > 0 && args[0].equalsIgnoreCase("Q")) {        	
        	destination = session.createQueue("Queue");	
        }else if(args.length > 0 && args[0].equalsIgnoreCase("T"))  {        	
        	destination = session.createTopic("Topic");        	
        }else {
        	System.out.println("Use Q for queue or T for topic");
        	connection.close();
        	System.exit(1);
        }
        
        // A MessageConsumer can only receive messages from a specific Topic or Queue.
        MessageConsumer consumer = session.createConsumer(destination);

        String body;
        do {
        	// The receive() method allows to consume messages from Topics or Queues.
         	// It pauses the application until a message is delivered from the Topic or Queue.
            Message msg = consumer.receive();
            body = ((TextMessage) msg).getText();
            System.out.println("Message received = " + body);
        }while (!body.equalsIgnoreCase("CLOSE"));
        
        connection.close();
        System.exit(1);
    }
}